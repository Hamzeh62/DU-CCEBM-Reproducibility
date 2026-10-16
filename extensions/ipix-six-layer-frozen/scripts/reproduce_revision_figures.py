"""Reproduce the three revision figures from the archived real-data run.

The script deliberately reconstructs the seeded data split used by the
revision drivers instead of inventing scores from the summary tables.  It
stores the score arrays and a small audit JSON alongside the PDFs so that a
future regeneration can be checked without rerunning training.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import du_ccebm_simulation as sim


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "outputs"
SNAP = OUT / "snapshots"
REAL_JSON = ROOT / "REVISION_REAL_RESULTS.json"
CKPT = ROOT / "checkpoints" / "real_A_best_model.pt"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def config_from_results(real: dict) -> dict:
    cfg = dict(sim.CFG)
    cfg.update(real["real_A_train_du_ccebm"]["config"])
    cfg.update(
        {
            "ipix_dir": str(ROOT / "ipix_data"),
            "save_dir": str(OUT / "real_du_ccebm"),
            "device": "cpu",
        }
    )
    return cfg


def complex_channels(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert (B,2,2,T) real/imag tensors to HH and VV complex arrays."""
    a = x.detach().cpu().numpy()
    hh = a[:, 0, 0] + 1j * a[:, 1, 0]
    vv = a[:, 0, 1] + 1j * a[:, 1, 1]
    return hh, vv


def build_mixed_split(cfg: dict, n_use: int, seed: int):
    sim.set_seed(seed)
    full = sim.IPIXDataset(cfg, mode="train")
    np.random.seed(seed)
    idx = np.random.choice(len(full), size=n_use, replace=False)
    subset = torch.utils.data.Subset(full, idx)
    n = len(subset)
    n_test = int(n * cfg["test_frac"])
    n_val = int(n * cfg["val_frac"])
    n_train = n - n_val - n_test
    _, _, test_set = torch.utils.data.random_split(
        subset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    loader = torch.utils.data.DataLoader(test_set, batch_size=cfg["batch"], shuffle=False)
    xs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            xs.append(x)
            ys.append(y)
    return full, subset, test_set, torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def load_model(cfg: dict):
    # weights_only=True avoids unpickling arbitrary objects in an archived
    # checkpoint while retaining the tensor state_dict and history metadata.
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = sim.DUCCEBM(
        L_net=cfg["L_net"], N=cfg["N"], L_filt=cfg["L_filt"], K_white=cfg["K_white"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def du_scores(model, x: torch.Tensor, batch: int = 512) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            scores.append(model(x[start : start + batch])[0].cpu().numpy())
    return np.concatenate(scores)


def common_scores(model, x: torch.Tensor, y: torch.Tensor, ccebm: bool):
    hh, vv = complex_channels(x)
    labels = y.cpu().numpy().astype(np.int64)
    scores_du = du_scores(model, x)

    clutter_idx = np.where(labels == 0)[0]
    clutter_pool = [np.stack([hh[i], vv[i]]) for i in clutter_idx]
    _, r_inv = sim.estimate_polarimetric_covariance(
        clutter_pool, n_calib=min(200, len(clutter_pool)), diag_loading=1e-2, seed=42
    )
    scores_anmf = sim.anmf_detect_batch(np.stack([hh, vv], axis=1), r_inv)

    amp = np.abs(hh).mean(axis=-1)
    scores_cfar = (amp - amp.min()) / (amp.max() - amp.min() + 1e-10)

    result = {
        "labels": labels,
        "DU-CCEBM": scores_du,
        "ANMF": scores_anmf,
        "CA-CFAR": scores_cfar,
    }

    if ccebm:
        classical = sim.ClassicalCCEBM(max_iter=50)
        ref_mean, ref_std = sim.calibrate_ccebm_reference(
            classical, clutter_pool, n_calib=min(50, len(clutter_pool)), seed=42
        )
        scores_cc = []
        for i in range(len(labels)):
            _, z2, _ = classical.fit_separate(hh[i], vv[i])
            nu2, _ = sim.k_distribution_mom(z2)
            scores_cc.append(sim.ccebm_calibrated_score(nu2, ref_mean, ref_std))
        result["Classical CCEBM"] = np.asarray(scores_cc, dtype=float)
        result["ccebm_ref"] = np.asarray([ref_mean, ref_std], dtype=float)

    return result


def clean_genuine_pool(cfg: dict, full, real: dict):
    """Reproduce the leak-free mask in run_real_B2_leak_fix.py."""
    train_cfg = real["real_A_train_du_ccebm"]
    n_full = len(full)
    if n_full != train_cfg["n_samples_loaded_total"]:
        raise RuntimeError(f"dataset size changed: {n_full} != {train_cfg['n_samples_loaded_total']}")

    np.random.seed(42)
    idx = np.random.choice(n_full, size=train_cfg["n_samples_used_subset"], replace=False)
    n = len(idx)
    n_test = int(n * cfg["test_frac"])
    n_val = int(n * cfg["val_frac"])
    n_train = n - n_val - n_test
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_pos = perm[:n_train]
    val_pos = perm[n_train : n_train + n_val]
    touched = set(idx[train_pos].tolist()) | set(idx[val_pos].tolist())

    # _load_ipix appends all genuine samples before synthetic injections.
    n_genuine = int(real["real_B_anmf_evaluation"]["genuine_only_full_pool"]["n"])
    samples = full.samples[:n_genuine]
    labels = np.asarray(full.labels[:n_genuine], dtype=np.float32)
    mask = np.asarray([i not in touched for i in range(n_genuine)], dtype=bool)
    samples = [samples[i] for i in np.flatnonzero(mask)]
    labels = labels[mask]
    hh = np.stack([s[0] for s in samples])
    vv = np.stack([s[1] for s in samples])
    x = np.zeros((len(samples), 2, 2, cfg["T"]), dtype=np.float32)
    x[:, 0, 0] = hh.real
    x[:, 1, 0] = hh.imag
    x[:, 0, 1] = vv.real
    x[:, 1, 1] = vv.imag
    return torch.from_numpy(x), labels, int(mask.size - mask.sum())


def plot_roc(score_dict: dict[str, np.ndarray], path: Path, title: str):
    colors = {
        "DU-CCEBM": "#1565C0",
        "ANMF": "#C62828",
        "CA-CFAR": "#2E7D32",
        "Classical CCEBM": "#6A1B9A",
    }
    fig, ax = plt.subplots(figsize=(3.55, 2.75))
    for name, scores in score_dict.items():
        if name in {"labels", "ccebm_ref"}:
            continue
        labels = score_dict["labels"]
        fpr, tpr, _ = roc_curve(labels, scores)
        value = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            lw=1.55,
            color=colors.get(name, "#333333"),
            label=f"{name} (AUC={value:.3f})",
        )
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=0.9, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-alarm rate")
    ax.set_ylabel("Detection probability")
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.grid(True, alpha=0.22, linewidth=0.45)
    ax.legend(loc="lower right", fontsize=5.8, frameon=True, framealpha=0.92)
    fig.tight_layout(pad=0.45)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history: dict, path: Path):
    epochs = np.arange(1, len(history["loss"]) + 1)
    # The paper inserts this graphic at one-column width, so vertically
    # stacked panels preserve readable labels after LaTeX scaling.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.55, 4.6))
    ax1.plot(epochs, history["loss"], label=r"$\mathcal{L}$", lw=1.4, color="#1565C0")
    ax1.plot(epochs, history["bce"], label=r"$\mathcal{L}_{det}$", lw=1.25, color="#C62828")
    ax1.plot(epochs, history["sep"], label=r"$\mathcal{L}_{sep}$", lw=1.25, color="#2E7D32")
    ax1.plot(epochs, history["pu"], label=r"$\mathcal{L}_{pu}$", lw=1.1, color="#6A1B9A")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss value")
    ax1.set_title("Training losses", fontsize=8.5, pad=4)
    ax1.grid(True, alpha=0.22, linewidth=0.45)
    ax1.legend(fontsize=6.2, frameon=True, framealpha=0.92)

    ax2.plot(epochs, history["auc_val"], marker="o", ms=2.5, lw=1.4, label="Validation AUC", color="#1565C0")
    ax2.plot(epochs, history["pd_val"], marker="s", ms=2.5, lw=1.25, label=r"$P_d$ @ $P_{fa}=10^{-3}$", color="#C62828")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_ylim(-0.01, max(0.66, max(history["auc_val"]) * 1.08))
    ax2.set_title("Validation metrics", fontsize=8.5, pad=4)
    ax2.grid(True, alpha=0.22, linewidth=0.45)
    ax2.legend(fontsize=6.2, frameon=True, framealpha=0.92)
    fig.tight_layout(pad=0.55, h_pad=0.9)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def rounded_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def main():
    OUT.mkdir(exist_ok=True)
    SNAP.mkdir(parents=True, exist_ok=True)
    real = load_json(REAL_JSON)
    cfg = config_from_results(real)
    model, ckpt = load_model(cfg)

    full, _, _, x_mix, y_mix = build_mixed_split(
        cfg, real["real_A_train_du_ccebm"]["n_samples_used_subset"], seed=42
    )
    mixed = common_scores(model, x_mix, y_mix, ccebm=True)

    x_clean, y_clean, n_excluded = clean_genuine_pool(cfg, full, real)
    clean = common_scores(model, x_clean, torch.from_numpy(y_clean), ccebm=False)

    history = ckpt["history"]
    plot_roc(mixed, PAPER / "fig2_roc.pdf", "Real IPIX mixed test split (n=240)")
    plot_roc(clean, PAPER / "fig6_roc_genuine_large.pdf", "Leak-free genuine-only pool (n=22,416)")
    plot_convergence(history, PAPER / "fig4_convergence.pdf")

    arrays_path = OUT / "revision_figures_scores.npz"
    np.savez_compressed(
        arrays_path,
        mixed_labels=mixed["labels"],
        mixed_du=mixed["DU-CCEBM"],
        mixed_anmf=mixed["ANMF"],
        mixed_cfar=mixed["CA-CFAR"],
        mixed_ccebm=mixed["Classical CCEBM"],
        clean_labels=clean["labels"],
        clean_du=clean["DU-CCEBM"],
        clean_anmf=clean["ANMF"],
        clean_cfar=clean["CA-CFAR"],
    )

    audit = {
        "seed": 42,
        "checkpoint": str(CKPT.relative_to(ROOT)),
        "mixed_n": int(len(mixed["labels"])),
        "clean_n": int(len(clean["labels"])),
        "clean_excluded_train_val": n_excluded,
        "mixed_auc": {k: rounded_auc(v, mixed["labels"]) for k, v in mixed.items() if k not in {"labels", "ccebm_ref"}},
        "clean_auc": {k: rounded_auc(v, clean["labels"]) for k, v in clean.items() if k != "labels"},
        "history_lengths": {k: len(v) for k, v in history.items()},
        "json_reference": {
            "mixed": real["real_B_anmf_evaluation"]["mixed_test_split"],
            "clean": real["real_B2_genuine_only_LEAK_FREE"]["result"],
        },
    }
    audit_path = OUT / "revision_figures_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    for name in ("fig2_roc.pdf", "fig4_convergence.pdf", "fig6_roc_genuine_large.pdf"):
        shutil.copy2(PAPER / name, SNAP / name)
    shutil.copy2(arrays_path, SNAP / arrays_path.name)
    shutil.copy2(audit_path, SNAP / audit_path.name)

    print(json.dumps({
        "mixed_auc": audit["mixed_auc"],
        "clean_auc": audit["clean_auc"],
        "mixed_n": audit["mixed_n"],
        "clean_n": audit["clean_n"],
        "clean_excluded_train_val": audit["clean_excluded_train_val"],
        "outputs": [str(PAPER / n) for n in ("fig2_roc.pdf", "fig4_convergence.pdf", "fig6_roc_genuine_large.pdf")],
    }, indent=2))


if __name__ == "__main__":
    main()
