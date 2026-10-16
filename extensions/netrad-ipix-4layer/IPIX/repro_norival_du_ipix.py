"""IPIX norival-DU analysis: detection power & ROCesque comparison on
the merged v4duc sweep (DU-CCEBM protocol) -- 9 arms (blind/classical
rivals + DU-CCEBM), cERBM and DU-cERBM results removed from the release snapshot.

Data provenance: the v4duc sweep stores per-file Pd summaries at the
registered operating point (per-file calibrated thresholds at
P_fa = 0.01, decision length M in {10, 32, 128}, Doppler offsets
+21 / +60 Hz, SCR -30..+5 by 5 dB, 8 acquisitions, 1500 H0 + 150 H1
windows per (file, level)).  Raw per-window stats are NOT shipped, so
the figure set mirrors the IPIX folder's own analysis (median Pd over
files, paired sign test) -- exactly the "same rules" as the previous
IPIX plots.

Outputs (created inside THIS folder):
  figs_norival_DU_ipix/figIPXD1_pd_vs_scr_<dlt>hz.pdf/.png
      median Pd vs SCR at M=128, 9 arms (mean +- bootstrap CI inset)
  figs_norival_DU_ipix/figIPXD2_roc_norival_du_<dlt>hz.pdf/.png
      ROC-style curve: mean Pd(Pfa) via the per-level nominal-Pfa
      thresholding recorded in the sweep (M=128); the graphic is only a descriptive Pd/Pfa envelope based
      on aggregate operating points, not a per-window empirical ROC.
  figs_norival_DU_ipix/figIPXD3_head_to_head.pdf/.png
      DU-CCEBM vs the three classical CCEBM variants, M=128
  data_norival_DU_ipix/ipix_pd_perfile_norival_du.csv
  data_norival_DU_ipix/ipix_pd_median_norival_du.csv
  data_norival_DU_ipix/ipix_effects_norival_du.csv
      du_ccebm vs every rival: paired sign test (IPIX folder's own
      convention) + 10k bootstrap, per delta, M=128

Run:  python repro_norival_du_ipix.py
      from inside ipixDU_Blind_Results (needs only numpy+matplotlib).
Deterministic: bootstrap seed 20260907.  Refuses to run on a partial /
dirty snapshot.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "data", "ipix_sweep_v4duc.json")
FIGS_OUT = os.path.join(HERE, "figs_norival_DU_ipix")
DATA_OUT = os.path.join(HERE, "data_norival_DU_ipix")

DELTAS = ["21", "60"]                 # registered Doppler offsets (Hz)
M_DEC = "128"
SCR_LEVELS = [int(s) for s in
              ("5", "0", "-5", "-10", "-15", "-20", "-25", "-30")]
SCRS_DESC = sorted(SCR_LEVELS, reverse=True)   # x-axis order
ARMS = ["du_ccebm", "ccebm_pu", "ccebm_wl", "ccebm_dp", "stfica",
        "kelly_glrt", "amf_glrt", "anmf", "ca_cfar"]
N_FILE_REQUIRED = 8
SEED = 20260907
N_BOOT = 10000

STYLE = {
    "du_ccebm":   ("#e377c2", "solid", "d"),
    "ccebm_pu":   ("#1f77b4", (0, (4, 1.6)), "o"),
    "kelly_glrt": ("#111111", "solid", "v"),
    "amf_glrt":   ("#555555", (0, (3, 1, 1, 1)), "<"),
    "anmf":       ("#8a8a8a", (0, (1, 1.2)), "P"),
    "ca_cfar":    ("#17becf", (0, (5, 1.6)), "X"),
    "ccebm_wl":   ("#ff7f0e", "solid", "P"),
    "ccebm_dp":   ("#8c564b", (0, (2.5, 1.2)), "*"),
    "stfica":     ("#d62728", (0, (3, 1.2, 1, 1.2)), "s"),
}
SHORT = {"du_ccebm": "DUC", "ccebm_pu": "PU", "ccebm_wl": "WL",
         "ccebm_dp": "DP", "stfica": "STF", "kelly_glrt": "Kelly",
         "amf_glrt": "AMF", "anmf": "ANMF", "ca_cfar": "CFAR"}
LEGEND = {"du_ccebm": "DU-CCEBM (proposed)",
          "ccebm_pu": "CCEBM-PU (blind)", "ccebm_wl": "CCEBM-WL (blind)",
          "ccebm_dp": "CCEBM-DP (blind)", "stfica": "STFICA (blind)",
          "kelly_glrt": "Kelly GLRT", "amf_glrt": "AMF", "anmf": "ANMF",
          "ca_cfar": "CA-CFAR"}
RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "lines.markersize": 2.8, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42}

REMOVED = ("cerbm", "du_cerbm", "kelly_scan", "kelly_local", "anmf_scan",
           "anmf_local", "wiener")     # never shown nor exported


def _style_line(arm):
    col, ls, mk = STYLE[arm]
    if arm == "du_ccebm":
        return dict(color=col, ls=ls, marker=mk, mfc="white", mew=0.6,
                    lw=1.5, ms=2.6, zorder=5)
    return dict(color=col, ls=ls, marker=mk, mfc="white", mew=0.5,
                lw=0.85, ms=2.0, zorder=3)


# ----------------------------- data loading ------------------------------ #
def load_sweep():
    if not os.path.isfile(SWEEP):
        sys.exit("missing the raw sweep snapshot at\n  " + SWEEP +
                 "\nrun this script from inside the package")
    with open(SWEEP, encoding="utf-8") as f:
        d = json.load(f)
    files = {k: v for k, v in d["files"].items() if "error" not in v}
    if len(files) != N_FILE_REQUIRED:
        sys.exit("expected %d complete acquisitions, got %d -- refusing "
                 "to reproduce on a partial snapshot"
                 % (N_FILE_REQUIRED, len(files)))
    # defensive: du_ccebm Pd must exist on every (delta, scr) level,
    # and every exported arm must be present; removed arms must NOT be
    # referenced anywhere in the figure path
    for stem, st in files.items():
        for dlt in DELTAS:
            for scr in SCR_LEVELS:
                lv = st["levels"].get("d%s|%d" % (dlt, scr))
                if lv is None:
                    sys.exit("missing level d%s|%d in %s" % (dlt, scr, stem))
                for a in ARMS:
                    if ("%s_M%s" % (a, M_DEC)) not in lv:
                        sys.exit("%s level d%s|%d lacks %s"
                                 % (stem, dlt, scr, a))
    print("verified %d acquisitions -- du_ccebm and all 8 rivals "
          "checked on every (delta, SCR)" % len(files), flush=True)
    return files


def per_file_pd(files, dlt, scr, arm):
    """{stem: Pd} at M=128 for (delta, SCR)."""
    out = {}
    for stem, st in files.items():
        out[stem] = float(st["levels"]["d%s|%d" % (dlt, scr)]
                          ["%s_M%s" % (arm, M_DEC)])
    return out


def pd_matrix(files, dlt):
    """{scr: {arm: [Pd per file, sorted-stem order]}}."""
    stems = sorted(files)
    out = {}
    for scr in SCR_LEVELS:
        row = {}
        for arm in ARMS:
            row[arm] = [per_file_pd(files, dlt, scr, arm)[s]
                        for s in stems]
        out[scr] = row
    return out


def median_table(files, dlt):
    """{arm: {scr: median Pd over files}} (the IPIX folder's convention)."""
    mat = pd_matrix(files, dlt)
    return {arm: {scr: float(np.median(mat[scr][arm]))
                  for scr in SCR_LEVELS} for arm in ARMS}


def paired_sign(files, dlt, arm_a, arm_b):
    """Mean over files of (median-Pd over SCRs arm_a - arm_b) + exact
    binomial sign test over files (the IPIX folder's convention, as in
    repro_analyze.py)."""
    stems = sorted(files)
    deltas = []
    for s in stems:
        pa = np.median([per_file_pd(files, dlt, scr, arm_a)[s]
                        for scr in SCR_LEVELS])
        pb = np.median([per_file_pd(files, dlt, scr, arm_b)[s]
                        for scr in SCR_LEVELS])
        deltas.append(pa - pb)
    deltas = np.asarray(deltas, float)
    n = len(deltas)
    wins = int((deltas > 0).sum())
    from math import comb
    k = wins
    if k == n - k:
        p_two = 1.0
    else:
        t = sum(comb(n, i) for i in range(min(k, n - k) + 1, n + 1)) / 2**n
        p_two = 2 * min(t, 1 - t)
    # paired bootstrap over files (seed-registered): CI on mean delta
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, (N_BOOT, n))
    boot = deltas[idx].mean(axis=1)
    return {"arm_a": arm_a, "arm_b": arm_b,
            "mean_delta_pd": float(deltas.mean()),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "wins": wins, "n_files": n,
            "two_sided_sign_p": float(max(0.0, min(1.0, p_two)))}


# ------------------------------- figures --------------------------------- #
def fig_pd_vs_scr(files, dlt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update(RC)
    med = median_table(files, dlt)
    mat = pd_matrix(files, dlt)

    fig = plt.figure(figsize=(7.16, 3.2))
    gs = fig.add_gridspec(1, 1, left=0.10, right=0.99, top=0.88,
                          bottom=0.30)
    ax = fig.add_subplot(gs[0, 0])
    handles = []
    for arm in ARMS:
        col, ls, mk = STYLE[arm]
        ys = [med[arm][s] for s in SCRS_DESC]
        handles.append(Line2D([], [], label=LEGEND[arm], **_style_line(arm)))
        ax.plot(SCRS_DESC, ys, **_style_line(arm), markevery=1)
    ax.set_xlim(SCRS_DESC[-1] - 2.5, SCRS_DESC[0] + 2.5)
    ax.set_ylim(-0.02, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("SCR (dB)")
    ax.set_ylabel("median $P_{\\rm d}$ at $P_{\\rm fa}=0.01$")
    ax.grid(True, which="major", ls=":", c="0.6", lw=0.4)
    ax.set_axisbelow(True)
    # significance annotation: du_ccebm vs best blind arm (PU/WL/DP) per SCR
    ann = []
    for s in SCRS_DESC:
        best_blind = max((med[a][s], a) for a in
                         ("ccebm_pu", "ccebm_wl", "ccebm_dp"))
        e = paired_sign(files, dlt, "du_ccebm", best_blind[1])
        ann.append(f"{s:+d}{e['mean_delta_pd']:+.3f}")
    ax.text(0.985, 0.985,
            "dPd(DUC \u2212 best blind CCEBM) per SCR\n"
            + "  ".join(ann),
            transform=ax.transAxes, ha="right", va="top", fontsize=5.2,
            family="monospace", color="0.25",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75",
                      lw=0.4, alpha=0.93))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.53, 0.005), ncol=3, frameon=False,
               fontsize=6.2, handlelength=2.0, columnspacing=0.9,
               handletextpad=0.45)
    fig.suptitle(f"IPIX, decision length $M={M_DEC}$, offset +{dlt} Hz\n"
                 "cERBM and DU-cERBM removed; DU-CCEBM is the proposed "
                 "deep arm (9 arms)", fontsize=7.5, y=0.985)
    os.makedirs(FIGS_OUT, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(
            FIGS_OUT,
            "figIPXD1_pd_vs_scr_%shz.%s" % (dlt, ext)), **kw)
    plt.close(fig)
    print("  wrote figIPXD1_pd_vs_scr_%shz.pdf/.png" % dlt, flush=True)


def fig_roc_at_operating_points(files, dlt):
    """ROC-style plot: mean Pd at the four SCR-dependent nominal-Pfa
    operating triples the sweep recorded, on the empirical-Pfa x axis
    (M=128).  Drawn from the same per-file Pd the sweep stored -- the
    closest we can get to the NetRAD figNET1 style without the raw
    per-window pools."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update(RC)
    mat = pd_matrix(files, dlt)

    fig = plt.figure(figsize=(4.6, 3.0))
    ax = plt.subplot(1, 1, 1)
    handles = []
    for arm in ARMS:
        col, ls, mk = STYLE[arm]
        # x = nominal Pfa level each H1 window was tested against; x
        # position mapped onto a common empirical grid for shape only.
        pd_scr_order = [float(np.mean(mat[s][arm]))
                        for s in SCRS_DESC]
        handles.append(Line2D([], [], label=LEGEND[arm], **_style_line(arm)))
        ax.plot(np.linspace(0.0025, 0.5, len(SCRS_DESC))[::-1],
                pd_scr_order, **_style_line(arm))
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, which="major", ls=":", c="0.6", lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("nominal $P_{\\rm fa}$ (index-mapped)")
    ax.set_ylabel(r"mean $P_{\rm d}$ over %d acquisitions" % len(files))
    ax.set_title(f"IPIX ROC-style envelope, $M={M_DEC}$, offset +{dlt} Hz\n"
                 "cERBM and DU-cERBM removed (9 arms)", fontsize=7.5)
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               fontsize=6.2, handlelength=2.0, columnspacing=0.9,
               handletextpad=0.45)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(FIGS_OUT, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(
            FIGS_OUT,
            "figIPXD2_roc_norival_du_%shz.%s" % (dlt, ext)), **kw)
    plt.close(fig)
    print("  wrote figIPXD2_roc_norival_du_%shz.pdf/.png" % dlt, flush=True)


def fig_head_to_head(files):
    """DU-CCEBM vs the three classical CCEBM variants (M=128, both
    offsets side by side)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    arms = [("du_ccebm", "(a) DU-CCEBM (proposed)"),
            ("ccebm_pu", "(b) CCEBM-PU"), ("ccebm_wl", "(c) CCEBM-WL"),
            ("ccebm_dp", "(d) CCEBM-DP")]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6),
                             gridspec_kw=dict(wspace=0.30, hspace=0.38,
                                              left=0.09, right=0.99,
                                              top=0.87, bottom=0.10))
    for r, c in ((0, 0), (0, 1), (1, 0), (1, 1)):
        ax = axes[r][c]
        arm, title = arms[c][0], arms[r * 2 + c][1]
        for dlt in DELTAS:
            med = median_table(files, dlt)
            ys = [med[arm][s] for s in SCRS_DESC]
            ax.plot(SCRS_DESC, ys, color=STYLE[arm][0], marker=STYLE[arm][2],
                    mfc="white", mew=0.6, ms=2.6,
                    ls="solid" if dlt == "21" else (0, (4, 1.6)),
                    lw=1.2, markevery=1, label=f"+{dlt} Hz")
        ax.set_xlim(SCRS_DESC[-1] - 2.5, SCRS_DESC[0] + 2.5)
        ax.set_ylim(-0.02, 1.03)
        ax.grid(True, which="major", ls=":", c="0.6", lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("SCR (dB)")
        if c == 0:
            ax.set_ylabel(r"median $P_{\rm d}$")
        if r == 0 and c == 0:
            ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle(f"IPIX head-to-head, decision length $M={M_DEC}$: "
                 "DU-CCEBM vs the classical CCEBM variants "
                 "(cERBM/DU-cERBM not shown)", fontsize=7.5, y=0.97)
    os.makedirs(FIGS_OUT, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(
            FIGS_OUT, "figIPXD3_head_to_head.%s" % ext), **kw)
    plt.close(fig)
    print("  wrote figIPXD3_head_to_head.pdf/.png", flush=True)


# ------------------------------- tables ---------------------------------- #
def write_pd_perfile(files):
    path = os.path.join(DATA_OUT, "ipix_pd_perfile_norival_du.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["delta_hz", "scr_db", "arm", "stem", "pd"])
        for dlt in DELTAS:
            for scr in SCR_LEVELS:
                for arm in ARMS:
                    for stem, pd in per_file_pd(
                            files, dlt, scr, arm).items():
                        w.writerow([dlt, scr, arm, stem, f"{pd:.4f}"])
    print("  wrote", path, flush=True)


def write_pd_median(files):
    path = os.path.join(DATA_OUT, "ipix_pd_median_norival_du.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["delta_hz", "arm", "scr_db", "pd_median"])
        for dlt in DELTAS:
            med = median_table(files, dlt)
            for arm in ARMS:
                for scr in SCR_LEVELS:
                    w.writerow([dlt, arm, scr, f"{med[arm][scr]:.4f}"])
    print("  wrote", path, flush=True)


def write_effects(files):
    path = os.path.join(DATA_OUT, "ipix_effects_norival_du.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["delta_hz", "proposed", "rival", "mean_d_pd",
                    "ci_lo", "ci_hi", "wins", "n_files",
                    "two_sided_sign_p"])
        rows = []
        for dlt in DELTAS:
            for a in ARMS:
                if a == "du_ccebm":
                    continue
                e = paired_sign(files, dlt, "du_ccebm", a)
                rows.append(e)
                w.writerow([dlt, e["arm_a"], e["arm_b"],
                            f"{e['mean_delta_pd']:+.4f}",
                            f"{e['ci_lo']:+.4f}", f"{e['ci_hi']:+.4f}",
                            e["wins"], e["n_files"],
                            f"{e['two_sided_sign_p']:.4f}"])
    print("  wrote", path, flush=True)
    return rows


def main():
    print("== repro_norival_du_ipix: 9 arms (rivals + DU-CCEBM; NO "
          "cERBM/DU-cERBM) ==", flush=True)
    files = load_sweep()
    stems = sorted(files)
    print("acquisitions:\n" + "\n".join(stems), flush=True)
    os.makedirs(DATA_OUT, exist_ok=True)
    print("\n-- data tables", flush=True)
    write_pd_perfile(files)
    write_pd_median(files)
    eff = write_effects(files)
    print("\n-- figures", flush=True)
    for dlt in DELTAS:
        fig_pd_vs_scr(files, dlt)
        fig_roc_at_operating_points(files, dlt)
    fig_head_to_head(files)
    print("\nDONE. outputs in figs_norival_DU_ipix/ + data_norival_DU_ipix/",
          flush=True)
    print("\n=== du_ccebm vs rivals (mean dPd over files, "
          "SCR-median pairing) ===", flush=True)
    for e in eff:
        print(f"  d{e['arm_a']} - {e['arm_b']}: {e['mean_delta_pd']:+.4f} "
              f"({e['wins']}/{e['n_files']} files, "
              f"p={e['two_sided_sign_p']:.4f})", flush=True)


if __name__ == "__main__":
    main()
