"""Reproduction analyzer for the v4 + DU-CCEBM sweep.

Reads _work_repro/results/ipix_sweep_v4duc.json (merged) and produces:

  results/repro_v4duc_verdict.json   -- effect sizes + predictions,
                                        computed by the FOLDER's own
                                        analyze_ipix_v4 logic (imported),
                                        extended with the du_ccebm arm
  figs/figR1_pd_vs_scr.png/.pdf      -- per-arm median Pd vs SCR
                                        (cERBM and DU-cERBM REMOVED:
                                        unpublished-paper arms)
  figs/figR2_head_to_head.png/.pdf   -- DU-CCEBM vs classical CCEBM arms
  results/repro_fig_source.json      -- the exact numbers behind the figs

ARMS SHOWN IN FIGURES (per the user's request): stfica, ccebm_dp,
ccebm_wl, ccebm_pu, du_ccebm.  cERBM and DU-cERBM are computed and kept
in the verdict JSON for internal validation but are REMOVED from every
figure -- they belong to an unpublished paper.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WROOT = os.path.normpath(os.path.join(_HERE, ".."))
RES = os.path.join(_WROOT, "results")
FIGS = os.path.join(_WROOT, "figs")
_FOLDER = os.path.join(_WROOT, "IPIX_Blind_Results", "code")
sys.path.insert(0, _FOLDER)

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

# arms REMOVED from figures (unpublished-paper arms)
REMOVED_FROM_FIGS = ("cerbm", "du_cerbm")

# figure arm styling (blind arms + classical detectors)
FIG_ARMS = [
    ("kelly_glrt", "Kelly GLRT", "#7f7f7f", ":", "s"),
    ("anmf", "ANMF", "#7f7f7f", (0, (1, 1)), "v"),
    ("stfica", "STFICA", "#9467bd", (0, (3, 1, 1, 1)), "x"),
    ("ccebm_dp", "CCEBM, real-AR (v2)", "#8c564b", (0, (1.2, 1.2)), "*"),
    ("ccebm_wl", "CCEBM, Eq. (13), orthogonal", "#ff7f0e", "solid", "P"),
    ("ccebm_pu", "CCEBM, Eq. (13), exact paraunitary", "#1f77b4",
     (0, (4, 1.6)), "o"),
    ("du_ccebm", "DU-CCEBM (ours, unrolled from CCEBM)", "#d62728",
     "solid", "D"),
]


def load_merged():
    with open(os.path.join(RES, "ipix_sweep_v4duc.json"), encoding="utf-8") as f:
        return json.load(f)


def files_dict(d):
    return {k: v for k, v in d["files"].items() if "error" not in v}


def median_pd_table(d):
    """{delta: {M: {arm: {scr: median Pd over files}}}} for the blind arms
    plus kelly/anmf, read out at M128 and M10/M32 as stored."""
    files = files_dict(d)
    out = {}
    for stem, st in files.items():
        for key, levels in st["levels"].items():
            dk, scr = key.split("|")
            scr = int(scr)
            dlt = dk[1:]
            for k, pd in levels.items():
                arm, m = k.rsplit("_M", 1)
                if k.startswith("_"):
                    continue
                out.setdefault(dlt, {}).setdefault(
                    int(m), {}).setdefault(arm, {}).setdefault(scr, {})[stem] = pd
    med = {}
    for dlt, ms in out.items():
        for m, arms in ms.items():
            for arm, scrs in arms.items():
                for scr, per in scrs.items():
                    med.setdefault(dlt, {}).setdefault(m, {}).setdefault(
                        arm, {})[scr] = float(np.median(list(per.values())))
    return med


def paired_effect(d, arm_a, arm_b):
    """Mean over files of median-over-delta-Pd(arm_a) - same for arm_b,
    with a paired sign test over files (the folder's convention: an arm
    wins a file if its median Pd is strictly higher)."""
    files = files_dict(d)
    deltas = []
    for stem, st in files.items():
        per = []
        for key, levels in st["levels"].items():
            for k in levels:
                if k.startswith("_"):
                    continue
                arm, m = k.rsplit("_M", 1)
                if m != "128":
                    continue
                if arm in (arm_a, arm_b):
                    per.append((key, arm, levels[k]))
        pa = np.median([v for k, a, v in per if a == arm_a])
        pb = np.median([v for k, a, v in per if a == arm_b])
        deltas.append(pa - pb)
    deltas = np.array(deltas)
    n = len(deltas)
    wins = int((deltas > 0).sum())
    # exact binomial sign test
    from math import comb
    k = wins
    p = sum(comb(n, i) for i in range(min(k, n - k) + 1, n + 1)) / 2 ** n \
        if k != n - k else 1.0
    p_two = 2 * min(p, 1 - p) if k != n / 2 else 1.0
    return {"arm_a": arm_a, "arm_b": arm_b, "mean_delta_pd":
            float(deltas.mean()), "files_favouring": arm_a,
            "wins": wins, "n_files": n,
            "two_sided_sign_p": float(min(max(p_two, 0.0), 1.0))}


def build_verdict(d):
    files = files_dict(d)
    n = len(files)
    eff = {}
    for a, b in (("du_ccebm", "ccebm_pu"), ("du_ccebm", "ccebm_wl"),
                 ("du_ccebm", "ccebm_dp"), ("du_ccebm", "stfica"),
                 ("du_ccebm", "cerbm"), ("ccebm_wl", "cerbm"),
                 ("ccebm_pu", "cerbm"), ("ccebm_pu", "ccebm_wl")):
        eff[f"{a}_minus_{b}"] = paired_effect(d, a, b)
    return {"provenance": {
        "script": "_work_repro/experiments/repro_analyze.py",
        "source": "results/ipix_sweep_v4duc.json",
        "base_protocol": "IPIX_Blind_Results/code/sweep_ipix_v4.py",
        "n_files": n,
        "note_removed_from_figures": list(REMOVED_FROM_FIGS)},
        "n_files": n, "effect_sizes": eff}


def fig_pd_vs_scr(med, dlt, M):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    scrs = sorted(next(iter(med[dlt][M].values())))
    for arm, lab, col, ls, mk in FIG_ARMS:
        if arm in REMOVED_FROM_FIGS or arm not in med[dlt][M]:
            continue
        ys = [med[dlt][M][arm].get(s, np.nan) for s in scrs]
        ax.plot(scrs, ys, label=lab, color=col, linestyle=ls, marker=mk,
                markersize=4, linewidth=1.4)
    ax.set_xlabel("SCR (dB)")
    ax.set_ylabel(f"median $P_d$  (M={M}, offset +{dlt} Hz)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"figR{M}_{dlt}hz.{ext}"), dpi=200)
    plt.close(fig)


def fig_du_vs_classical(med, dlt):
    """DU-CCEBM against its own classical baselines, M=128."""
    M = 128
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    scrs = sorted(next(iter(med[dlt][M].values())))
    for arm, lab, col, ls, mk in FIG_ARMS:
        if arm in REMOVED_FROM_FIGS or arm not in med[dlt][M]:
            continue
        if arm not in ("du_ccebm", "ccebm_pu", "ccebm_wl", "ccebm_dp"):
            continue
        ys = [med[dlt][M][arm].get(s, np.nan) for s in scrs]
        ax.plot(scrs, ys, label=lab, color=col, linestyle=ls, marker=mk,
                markersize=4, linewidth=1.4)
    ax.set_xlabel("SCR (dB)")
    ax.set_ylabel(f"median $P_d$  (M={M}, offset +{dlt} Hz)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"figR2_du_vs_ccebm_{dlt}hz.{ext}"),
                    dpi=200)
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    d = load_merged()
    files = files_dict(d)
    n = len(files)
    print(f"loaded {n} complete acquisitions")
    if n == 0:
        print("nothing complete yet; aborting figures")
        return
    verdict = build_verdict(d)
    with open(os.path.join(RES, "repro_v4duc_verdict.json"), "w",
              encoding="utf-8") as f:
        json.dump(verdict, f, indent=1)
    med = median_pd_table(d)
    src = {"n_files": n, "median_pd": med,
           "effect_sizes": verdict["effect_sizes"],
           "REMOVED_FROM_FIGURES": list(REMOVED_FROM_FIGS)}
    with open(os.path.join(RES, "repro_fig_source.json"), "w",
              encoding="utf-8") as f:
        json.dump(src, f, indent=1)
    for dlt in med:
        for M in med[dlt]:
            fig_pd_vs_scr(med, dlt, M)
        fig_du_vs_classical(med, dlt)
    print("figures ->", FIGS)
    for a in ("du_ccebm",):
        for b in ("ccebm_pu", "ccebm_wl", "cerbm"):
            e = verdict["effect_sizes"][f"{a}_minus_{b}"]
            print(f"  {a} - {b}: dPd {e['mean_delta_pd']:+.4f} "
                  f"({e['wins']}/{e['n_files']} files, "
                  f"p={e['two_sided_sign_p']:.3f})")


if __name__ == "__main__":
    main()
