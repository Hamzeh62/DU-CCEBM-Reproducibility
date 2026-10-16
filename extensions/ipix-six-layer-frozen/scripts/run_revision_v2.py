"""Run the TAES revision-v2 experiments without changing trained weights.

The runner deliberately reuses the archived score cache for DU-CCEBM,
ANMF, and CA-CFAR.  Classical CCEBM is the only full-pool detector that is
newly evaluated.  Its scores are checkpointed every 2,000 windows so a
stopped run resumes at the first unfinished index.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import du_ccebm_simulation as sim
from classical_ref.optimized_ccebm import (
    NU2_SATURATION_THRESH,
    calibrate_reference,
    run_batched_ccebm,
)
from scripts.reproduce_revision_figures import (
    build_mixed_split,
    clean_genuine_pool,
    complex_channels,
    config_from_results,
    du_scores,
    load_json,
    load_model,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
SCORES_CACHE = OUTPUTS / "revision_figures_scores.npz"
OLD_JSON = ROOT / "REVISION_REAL_RESULTS.json"
NEW_JSON = RESULTS / "REVISION_REAL_RESULTS_v2.json"
CHECKPOINT = ROOT / "checkpoints" / "real_A_best_model.pt"
CCEBM_PROGRESS = RESULTS / "ccebm_full_scores_progress.npz"
FINAL_SCORES = RESULTS / "revision_v2_scores.npz"
BOOTSTRAP_PROGRESS = RESULTS / "bootstrap_block_progress.npz"
PARAM_CSV = RESULTS / "per_layer_params.csv"

SEED = 42
N_BOOT = 2000
CHECKPOINT_EVERY = 2000
PFA_VALUES = (1e-2, 1e-3)
METHODS = ("DU-CCEBM", "Classical CCEBM", "ANMF", "CA-CFAR")


def save_json(payload: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    NEW_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUTPUTS.mkdir(exist_ok=True)
    shutil.copy2(NEW_JSON, OUTPUTS / NEW_JSON.name)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def clean_file_ids(full, real: dict) -> np.ndarray:
    """Apply the exact leak-free mask to the genuine-file provenance."""
    train_cfg = real["real_A_train_du_ccebm"]
    n_genuine = int(real["real_B_anmf_evaluation"]["genuine_only_full_pool"]["n"])
    np.random.seed(SEED)
    idx = np.random.choice(
        len(full), size=train_cfg["n_samples_used_subset"], replace=False
    )
    n = len(idx)
    n_test = int(n * train_cfg["config"].get("test_frac", 0.10))
    n_val = int(n * train_cfg["config"].get("val_frac", 0.15))
    n_train = n - n_val - n_test
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
    touched = set(idx[perm[:n_train]].tolist()) | set(
        idx[perm[n_train : n_train + n_val]].tolist()
    )
    mask = np.asarray([i not in touched for i in range(n_genuine)], dtype=bool)
    return np.asarray(full.genuine_file_ids[:n_genuine], dtype=str)[mask]


def prepare_data():
    real = load_json(OLD_JSON)
    cfg = config_from_results(real)
    full, _, _, x_mixed, y_mixed = build_mixed_split(
        cfg, real["real_A_train_du_ccebm"]["n_samples_used_subset"], SEED
    )
    x_clean, y_clean, n_excluded = clean_genuine_pool(cfg, full, real)
    file_ids = clean_file_ids(full, real)
    if len(file_ids) != len(y_clean):
        raise RuntimeError("clean file-id mask does not align with clean labels")

    cache = np.load(SCORES_CACHE)
    if not np.array_equal(cache["clean_labels"], y_clean.astype(np.int64)):
        raise RuntimeError("cached clean-score labels no longer match reconstructed data")
    if not np.array_equal(cache["mixed_labels"], y_mixed.numpy().astype(np.int64)):
        raise RuntimeError("cached mixed-score labels no longer match reconstructed split")

    a = x_clean.numpy()
    clean_complex = np.stack(
        [a[:, 0, 0] + 1j * a[:, 1, 0], a[:, 0, 1] + 1j * a[:, 1, 1]],
        axis=1,
    )
    hm, vm = complex_channels(x_mixed)
    mixed_complex = np.stack([hm, vm], axis=1)
    return {
        "real": real,
        "cfg": cfg,
        "full": full,
        "x_clean": x_clean,
        "y_clean": y_clean.astype(np.int64),
        "file_ids": file_ids,
        "clean_complex": clean_complex,
        "x_mixed": x_mixed,
        "y_mixed": y_mixed.numpy().astype(np.int64),
        "mixed_complex": mixed_complex,
        "n_excluded": n_excluded,
        "cache": cache,
    }


def checkpointed_ccebm_scores(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compute or resume all full-pool classical scores."""
    solver_kwargs = dict(L=8, K=5, max_iter=50, tol=1e-5, prefer_cuda=True)
    ref_median, ref_mad, calibration = calibrate_reference(
        X[labels == 0], n_calib=50, seed=SEED, **solver_kwargs
    )

    n = len(X)
    nu2 = np.full(n, np.nan, dtype=np.float64)
    scores = np.full(n, np.nan, dtype=np.float64)
    iterations = np.zeros(n, dtype=np.int32)
    elapsed_completed = 0.0
    if CCEBM_PROGRESS.exists():
        old = np.load(CCEBM_PROGRESS)
        if int(old["n_total"]) == n and np.isclose(old["ref_median"], ref_median):
            nu2[:] = old["nu2"]
            scores[:] = old["scores"]
            iterations[:] = old["iterations"]
            elapsed_completed = float(old["elapsed_s"])
            print(f"[CCEBM] resumed {np.isfinite(scores).sum():,}/{n:,} scores", flush=True)

    first = int(np.flatnonzero(~np.isfinite(scores))[0]) if np.any(~np.isfinite(scores)) else n
    started = time.perf_counter()
    last_device = str(calibration["device"])
    n_jobs = int(calibration["n_jobs"])
    for start in range(first, n, CHECKPOINT_EVERY):
        stop = min(n, start + CHECKPOINT_EVERY)
        run = run_batched_ccebm(
            X[start:stop], ref_median, ref_mad, **solver_kwargs
        )
        scores[start:stop] = run.scores
        nu2[start:stop] = run.nu2
        iterations[start:stop] = run.iterations
        last_device, n_jobs = run.device, run.n_jobs
        elapsed = elapsed_completed + time.perf_counter() - started
        done = stop
        rate = done / max(elapsed, 1e-9)
        eta = (n - done) / max(rate, 1e-9)
        vm = psutil.virtual_memory()
        if vm.percent >= 80.0:
            raise MemoryError(f"system RAM reached {vm.percent:.1f}% (80% cap)")
        np.savez_compressed(
            CCEBM_PROGRESS,
            scores=scores,
            nu2=nu2,
            iterations=iterations,
            n_total=n,
            ref_median=ref_median,
            ref_mad=ref_mad,
            elapsed_s=elapsed,
            device=run.device,
            n_jobs=run.n_jobs,
        )
        print(
            f"[CCEBM] {done:,}/{n:,}; elapsed={elapsed:.1f}s; ETA={eta:.1f}s; "
            f"RAM={vm.percent:.1f}% ({vm.used / 2**30:.2f} GiB); "
            f"device={run.device}; n_jobs={run.n_jobs}",
            flush=True,
        )

    elapsed = (
        elapsed_completed
        if first == n
        else elapsed_completed + time.perf_counter() - started
    )
    legacy = load_json(OLD_JSON)["real_E_ccebm_iter_sensitivity_mixed_split"]
    legacy_50 = legacy["results_by_max_iter"]["50"]
    legacy_ms = 1000.0 * float(legacy_50["wall_clock_s"]) / int(legacy["n"])
    legacy_projected_s = legacy_ms * n / 1000.0
    audit = {
        "implementation": "vectorized batched torch/NumPy equivalent of ClassicalCCEBM",
        "solver": {"L": 8, "K": 5, "max_iter": 50, "tol": 1e-5},
        "device": last_device,
        "n_jobs": int(n_jobs),
        "wall_clock_s": float(elapsed),
        "mean_ms_per_window_including_checkpoints": float(1000.0 * elapsed / n),
        "target_under_5_min_met": bool(elapsed < 300.0),
        "legacy_scalar_mixed_split_ms_per_window": float(legacy_ms),
        "legacy_scalar_projected_full_pool_s": float(legacy_projected_s),
        "measured_speedup_vs_legacy_scalar_rate": float(legacy_projected_s / elapsed),
        "n_saturated": int(np.sum(nu2 >= NU2_SATURATION_THRESH)),
        "saturation_rate": float(np.mean(nu2 >= NU2_SATURATION_THRESH)),
        "mean_iterations": float(iterations.mean()),
        "reference_median": float(ref_median),
        "reference_mad": float(ref_mad),
        "calibration": calibration,
        "checkpoint_every_windows": CHECKPOINT_EVERY,
        "checkpoint_path": str(CCEBM_PROGRESS.relative_to(ROOT)),
    }
    return scores, audit


def _pd(scores: np.ndarray, labels: np.ndarray, pfa: float) -> float:
    return sim.compute_pd_at_pfa(scores, labels, pfa=pfa)


def _pauc(scores: np.ndarray, labels: np.ndarray, lo=1e-3, hi=1e-1) -> float:
    """Normalised AUC over the requested FAR interval [lo, hi]."""
    fpr, tpr, _ = roc_curve(labels, scores)
    grid = np.unique(np.r_[lo, fpr[(fpr > lo) & (fpr < hi)], hi])
    values = np.interp(grid, fpr, tpr)
    # NumPy 2.0 removed the legacy ``trapz`` alias; ``trapezoid`` is
    # equivalent and remains available on both current and older releases.
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(values, grid) / (hi - lo))


def _blocks(file_ids: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    names = sorted(np.unique(file_ids).tolist())
    return names, [np.flatnonzero(file_ids == name) for name in names]


def _block_sample_indices(
    rng: np.random.Generator, block_indices: list[np.ndarray]
) -> np.ndarray:
    selected = rng.integers(0, len(block_indices), size=len(block_indices))
    return np.concatenate([block_indices[int(i)] for i in selected])


def block_bootstrap(
    scores: dict[str, np.ndarray], labels: np.ndarray, file_ids: np.ndarray
) -> dict:
    """Resumable file-block bootstrap for E1 and E4, seed 42, 2,000 draws."""
    names, indices = _blocks(file_ids)
    metric_names = []
    for method in METHODS:
        metric_names.extend(
            [f"auc::{method}", f"pd1e2::{method}", f"pd1e3::{method}", f"pauc::{method}"]
        )
    metric_names.extend(
        [
            "delta::DU-CCEBM::Classical CCEBM",
            "delta::DU-CCEBM::CA-CFAR",
            "delta::Classical CCEBM::CA-CFAR",
        ]
    )
    values = np.full((N_BOOT, len(metric_names)), np.nan, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    completed = 0
    if BOOTSTRAP_PROGRESS.exists():
        old = np.load(BOOTSTRAP_PROGRESS, allow_pickle=True)
        old_names = old["metric_names"].tolist()
        if old_names == metric_names and int(old["n_boot"]) == N_BOOT:
            values[:] = old["values"]
            completed = int(old["completed"])
            rng.bit_generator.state = old["rng_state"].item()
            print(f"[bootstrap] resumed {completed:,}/{N_BOOT:,}", flush=True)

    for b in range(completed, N_BOOT):
        bi = _block_sample_indices(rng, indices)
        yb = labels[bi]
        row = []
        aucs = {}
        if np.unique(yb).size < 2:
            row = [np.nan] * len(metric_names)
        else:
            for method in METHODS:
                sb = scores[method][bi]
                aucs[method] = roc_auc_score(yb, sb)
                row.extend(
                    [
                        aucs[method],
                        _pd(sb, yb, 1e-2),
                        _pd(sb, yb, 1e-3),
                        _pauc(sb, yb),
                    ]
                )
            row.extend(
                [
                    aucs["DU-CCEBM"] - aucs["Classical CCEBM"],
                    aucs["DU-CCEBM"] - aucs["CA-CFAR"],
                    aucs["Classical CCEBM"] - aucs["CA-CFAR"],
                ]
            )
        values[b] = row
        if (b + 1) % 100 == 0 or b + 1 == N_BOOT:
            np.savez_compressed(
                BOOTSTRAP_PROGRESS,
                values=values,
                metric_names=np.asarray(metric_names),
                completed=b + 1,
                n_boot=N_BOOT,
                rng_state=np.asarray(rng.bit_generator.state, dtype=object),
            )
            print(f"[bootstrap] {b + 1:,}/{N_BOOT:,}", flush=True)

    def summary(name: str, point: float) -> dict:
        vals = values[:, metric_names.index(name)]
        vals = vals[np.isfinite(vals)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {
            "point": float(point),
            "ci95": [float(lo), float(hi)],
            "n_boot_valid": int(len(vals)),
        }

    out = {"design": {
        "type": "paired nonparametric bootstrap by IPIX file",
        "n_resamples": N_BOOT,
        "seed": SEED,
        "n_blocks": len(names),
        "blocks": names,
        "rng": "numpy.random.Generator (PCG64), state checkpointed every 100 draws",
    }}
    out["methods"] = {}
    for method in METHODS:
        out["methods"][method] = {
            "auc": summary(f"auc::{method}", roc_auc_score(labels, scores[method])),
            "pd_at_pfa_1e-2": summary(
                f"pd1e2::{method}", _pd(scores[method], labels, 1e-2)
            ),
            "pd_at_pfa_1e-3": summary(
                f"pd1e3::{method}", _pd(scores[method], labels, 1e-3)
            ),
            "normalized_auc_far_1e-3_to_1e-1": summary(
                f"pauc::{method}", _pauc(scores[method], labels)
            ),
        }
    out["paired_auc_differences"] = {}
    for a, b in (
        ("DU-CCEBM", "Classical CCEBM"),
        ("DU-CCEBM", "CA-CFAR"),
        ("Classical CCEBM", "CA-CFAR"),
    ):
        key = f"{a}_minus_{b}"
        entry = summary(
            f"delta::{a}::{b}",
            roc_auc_score(labels, scores[a]) - roc_auc_score(labels, scores[b]),
        )
        entry["statistically_significant_at_0.05"] = not (
            entry["ci95"][0] <= 0.0 <= entry["ci95"][1]
        )
        out["paired_auc_differences"][key] = entry
    return out


def benchmark_methods(data: dict, ccebm_ref: tuple[float, float]) -> dict:
    """Warm up 20, then time exactly 500 windows for all four methods."""
    X = data["clean_complex"][:500]
    x = data["x_clean"][:500]
    model, _ = load_model(data["cfg"])
    labels = data["y_clean"]
    clutter_pool = X[labels[: len(X)] == 0]
    _, r_inv = sim.estimate_polarimetric_covariance(
        list(clutter_pool), n_calib=min(200, len(clutter_pool)), seed=SEED
    )

    timings = {}

    def timed(name, warm_fn, run_fn):
        warm_fn()
        t0 = time.perf_counter()
        run_fn()
        elapsed = time.perf_counter() - t0
        timings[name] = {
            "n_windows": 500,
            "warmup_windows": 20,
            "wall_clock_s": float(elapsed),
            "mean_ms_per_window": float(1000.0 * elapsed / 500.0),
        }

    timed(
        "DU-CCEBM",
        lambda: du_scores(model, x[:20], batch=20),
        lambda: du_scores(model, x, batch=500),
    )
    timed(
        "ANMF",
        lambda: sim.anmf_detect_batch(X[:20], r_inv),
        lambda: sim.anmf_detect_batch(X, r_inv),
    )

    def cfar(arr):
        return np.abs(arr[:, 0]).mean(axis=-1)

    timed("CA-CFAR", lambda: cfar(X[:20]), lambda: cfar(X))
    timed(
        "Classical CCEBM",
        lambda: run_batched_ccebm(
            X[:20], *ccebm_ref, L=8, K=5, max_iter=50, tol=1e-5
        ),
        lambda: run_batched_ccebm(
            X, *ccebm_ref, L=8, K=5, max_iter=50, tol=1e-5
        ),
    )
    timings["environment"] = {
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpu_count": int(os.cpu_count() or 1),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    return timings


def frozen_lofo(scores: dict[str, np.ndarray], labels: np.ndarray, file_ids: np.ndarray) -> dict:
    """Frozen-checkpoint per-file evaluation, as allowed when retraining is deferred."""
    rows = []
    for name in sorted(np.unique(file_ids).tolist()):
        m = file_ids == name
        if np.unique(labels[m]).size < 2:
            raise RuntimeError(f"file {name} does not contain both classes")
        rows.append(
            {
                "held_out_file": name,
                "n": int(m.sum()),
                "n_target": int(labels[m].sum()),
                "n_clutter": int(m.sum() - labels[m].sum()),
                **{
                    f"auc_{method}": float(roc_auc_score(labels[m], values[m]))
                    for method, values in scores.items()
                },
            }
        )
    du = np.asarray([row["auc_DU-CCEBM"] for row in rows])
    return {
        "protocol": "file-stratified frozen-checkpoint evaluation (LOFO proxy)",
        "full_retraining_status": "not run; optional extension (trained weights unchanged)",
        "interpretation": (
            "The leak-free genuine windows are grouped by file and scored by the "
            "archived checkpoint. This is not strict leave-one-file-out training: "
            "the model was not retrained 14 times with each entire file excluded."
        ),
        "per_file": rows,
        "DU-CCEBM_mean_auc": float(du.mean()),
        "DU-CCEBM_std_auc": float(du.std(ddof=1)),
    }


def extract_layer_params(model) -> list[dict]:
    rows = []
    for layer_index, layer in enumerate(model.layers, start=1):
        beta_values, a_values = [], []
        for component, modules in (("R", layer.meas_R), ("I", layer.meas_I)):
            for source, module in enumerate(modules, start=1):
                beta = torch.softmax(module.logit_beta.detach().cpu(), dim=0).numpy()
                a_value = float(torch.clamp(module.a1.detach().cpu(), 0.5, 2.0).item())
                beta_values.append(beta)
                a_values.append(a_value)
                rows.append(
                    {
                        "layer": layer_index,
                        "source": source,
                        "component": component,
                        "beta1": float(beta[0]),
                        "beta2": float(beta[1]),
                        "a": a_value,
                        "tau": float(torch.exp(layer.log_tau.detach().cpu())[source - 1]),
                        "gamma": float(torch.sigmoid(layer.logit_gamma.detach().cpu()).item()),
                    }
                )
        betas = np.vstack(beta_values)
        rows.append(
            {
                "layer": layer_index,
                "source": "mean",
                "component": "mean",
                "beta1": float(betas[:, 0].mean()),
                "beta2": float(betas[:, 1].mean()),
                "a": float(np.mean(a_values)),
                "tau": float(torch.exp(layer.log_tau.detach().cpu()).mean()),
                "gamma": float(torch.sigmoid(layer.logit_gamma.detach().cpu()).item()),
            }
        )
    return rows


def save_params(rows: list[dict]) -> None:
    PARAM_CSV.parent.mkdir(exist_ok=True)
    with PARAM_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(PARAM_CSV, OUTPUTS / PARAM_CSV.name)


def plot_layer_heatmap(rows: list[dict]) -> None:
    mean_rows = [row for row in rows if row["component"] == "mean"]
    matrix = np.asarray(
        [[row[key] for key in ("beta1", "beta2", "a", "tau", "gamma")] for row in mean_rows]
    ).T
    labels = [r"$\beta_1$", r"$\beta_2$", r"$a$", r"$\tau$", r"$\gamma$"]
    fig, ax = plt.subplots(figsize=(4.4, 2.55))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(mean_rows)), [str(r["layer"]) for r in mean_rows])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Unrolled layer")
    ax.set_title("Learned layer-wise parameter means")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] < matrix.max() * 0.55 else "black"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout(pad=0.5)
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"layer_params_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_zoom(scores: dict[str, np.ndarray], labels: np.ndarray) -> None:
    colors = {
        "DU-CCEBM": "#1565C0",
        "Classical CCEBM": "#6A1B9A",
        "ANMF": "#C62828",
        "CA-CFAR": "#2E7D32",
    }
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    for method in METHODS:
        fpr, tpr, _ = roc_curve(labels, scores[method])
        ax.semilogx(fpr[fpr > 0], tpr[fpr > 0], label=method, color=colors[method], lw=1.5)
    ax.set_xlim(1e-3, 1e-1)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel(r"False-alarm probability $P_{fa}$")
    ax.set_ylabel(r"Detection probability $P_d$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout(pad=0.5)
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"roc_genuine_zoom.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    data = prepare_data()
    classical, ccebm_audit = checkpointed_ccebm_scores(
        data["clean_complex"], data["y_clean"]
    )
    clean_scores = {
        "DU-CCEBM": np.asarray(data["cache"]["clean_du"]),
        "Classical CCEBM": classical,
        "ANMF": np.asarray(data["cache"]["clean_anmf"]),
        "CA-CFAR": np.asarray(data["cache"]["clean_cfar"]),
    }
    np.savez_compressed(
        FINAL_SCORES,
        labels=data["y_clean"],
        file_ids=data["file_ids"],
        du=clean_scores["DU-CCEBM"],
        ccebm=clean_scores["Classical CCEBM"],
        anmf=clean_scores["ANMF"],
        ca_cfar=clean_scores["CA-CFAR"],
    )

    inference = block_bootstrap(clean_scores, data["y_clean"], data["file_ids"])
    e2 = benchmark_methods(
        data, (ccebm_audit["reference_median"], ccebm_audit["reference_mad"])
    )
    e3 = frozen_lofo(clean_scores, data["y_clean"], data["file_ids"])

    mixed_labels = data["y_mixed"]
    raw_anmf = np.asarray(data["cache"]["mixed_anmf"])
    auc_raw = float(roc_auc_score(mixed_labels, raw_anmf))
    auc_flipped = float(roc_auc_score(mixed_labels, -raw_anmf))
    e5 = {
        "n": int(len(mixed_labels)),
        "n_target": int(mixed_labels.sum()),
        "n_clutter": int(len(mixed_labels) - mixed_labels.sum()),
        "ANMF_auc_original": auc_raw,
        "ANMF_auc_sign_flipped": auc_flipped,
        "identity_check_one_minus_original": float(1.0 - auc_raw),
        "reported_for_fairness_only": True,
    }

    model, _ = load_model(data["cfg"])
    param_rows = extract_layer_params(model)
    save_params(param_rows)
    plot_layer_heatmap(param_rows)
    plot_roc_zoom(clean_scores, data["y_clean"])

    payload = {
        "schema_version": 2,
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "source_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "trained_weights_changed": False,
        "source_score_cache": str(SCORES_CACHE.relative_to(ROOT)),
        "pool": {
            "n": int(len(data["y_clean"])),
            "n_target": int(data["y_clean"].sum()),
            "n_clutter": int(len(data["y_clean"]) - data["y_clean"].sum()),
            "n_excluded_train_val_leakage": int(data["n_excluded"]),
            "n_files": int(len(np.unique(data["file_ids"]))),
            "T": int(data["cfg"]["T"]),
        },
        "E1_full_classical_and_paired_bootstrap": {
            "classical_run": ccebm_audit,
            "inference": inference,
        },
        "E2_wall_clock": e2,
        "E3_leave_one_file_out": e3,
        "E4_operating_points": {
            "far_range": [1e-3, 1e-1],
            "normalized_partial_auc_definition": "trapezoidal ROC area divided by 0.1-0.001",
            "methods": {
                method: {
                    "auc": inference["methods"][method]["auc"],
                    "pd_at_pfa_1e-2": inference["methods"][method]["pd_at_pfa_1e-2"],
                    "pd_at_pfa_1e-3": inference["methods"][method]["pd_at_pfa_1e-3"],
                    "normalized_auc_far_1e-3_to_1e-1": inference["methods"][method][
                        "normalized_auc_far_1e-3_to_1e-1"
                    ],
                }
                for method in METHODS
            },
        },
        "E5_mixed_sign_flipped_anmf": e5,
        "E6_layer_parameters": {
            "csv": str(PARAM_CSV.relative_to(ROOT)),
            "heatmap_pdf": "figures/layer_params_heatmap.pdf",
            "heatmap_png": "figures/layer_params_heatmap.png",
            "rows": param_rows,
        },
        "artifacts": {
            "scores": str(FINAL_SCORES.relative_to(ROOT)),
            "roc_zoom_pdf": "figures/roc_genuine_zoom.pdf",
            "roc_zoom_png": "figures/roc_genuine_zoom.png",
        },
    }
    save_json(jsonable(payload))
    for path in FIGURES.glob("*.pdf"):
        if path.name in {"roc_genuine_zoom.pdf", "layer_params_heatmap.pdf"}:
            shutil.copy2(path, OUTPUTS / path.name)
    for path in FIGURES.glob("*.png"):
        if path.name in {"roc_genuine_zoom.png", "layer_params_heatmap.png"}:
            shutil.copy2(path, OUTPUTS / path.name)
    print(json.dumps(payload["E1_full_classical_and_paired_bootstrap"]["inference"], indent=2))


if __name__ == "__main__":
    main()
