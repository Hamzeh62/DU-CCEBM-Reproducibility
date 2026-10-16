"""Standalone, GitHub-ready reproduction: the NetRAD Path-A ROC &
detection-power study with ONLY the blind/classical rival arms plus the
appended DU-CCEBM arm (9 arms) -- cERBM and DU-cERBM REMOVED, as
requested.

Reads ONLY the included raw checkpoint snapshot ``data/checkpoints_du/``
(contains the nine retained arms including DU-CCEBM; the two
excluded detector arms have been removed from this release snapshot), so no radar data and no network access are needed. Needs only
numpy + matplotlib.

Outputs (created inside THIS folder):
  figs_norival_DU/figNETN1D_roc_norival_du.pdf/.png      ROC per SCR, 9 arms
  figs_norival_DU/figNETN2D_power_norival_du.pdf/.png    P_d @ P_fa=0.01
  figs_norival_DU/figNETN3D_roc_peracq_norival_du.pdf/.png per-acq spread
  data_norival_DU/netrad_roc_curves_norival_du.csv        mean ROC curves
  data_norival_DU/netrad_auc_norival_du.csv               per-SCR per-arm AUC
  data_norival_DU/netrad_power_pfa01_norival_du.csv       P_d summary (95% CI)
  data_norival_DU/netrad_power_peracq_norival_du.csv      P_d per acquisition

Protocol invariants (identical to the registered run):
  * exactly 21 complete checkpoints
  * AUC = Mann-Whitney, ties counting 0.5, NaN-tolerant pairwise
  * ROC grid: 30 empirical-P_fa points, geomspace(0.0025, 0.5)
  * operating point: per-file calibrated threshold, P_fa = 0.01
  * power CI bootstrap: seed 20260907 + int(scr)  (registered scheme)
Arms kept: CCEBM-PU/WL/DP, STFICA, Kelly, AMF, ANMF, CA-CFAR (style
identical to the paper plots) + DU-CCEBM as the proposed deep arm.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_SNAPSHOT = os.path.join(HERE, "data", "checkpoints_du")
FIGS_OUT = os.path.join(HERE, "figs_norival_DU")
DATA_OUT = os.path.join(HERE, "data_norival_DU")

# the 9 arms kept: 8 rivals (style identical to the paper) + DU-CCEBM
ARMS = ["du_ccebm", "ccebm_pu", "ccebm_wl", "ccebm_dp", "stfica",
        "kelly_glrt", "amf_glrt", "anmf", "ca_cfar"]
SCR_LEVELS = [-25.0, -20.0, -15.0, -10.0, -5.0]
SEED = 20260907
N_BOOT = 10000
PFA_REG = 0.01
PFA_GRID = np.geomspace(0.0025, 0.5, 30)

# registered figure style (copied verbatim from the paper plots; the
# DU-CCEBM entry mirrors the appended-arm style in du_netroc_figs.py)
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

# du_proj highlight style for the proposed arm (thicker, on top)
def _style_line(arm):
    col, ls, mk = STYLE[arm]
    if arm == "du_ccebm":
        return dict(color=col, ls=ls, marker=mk, mfc="white", mew=0.6,
                    lw=1.5, ms=2.6, zorder=5)
    return dict(color=col, ls=ls, marker=mk, mfc="white", mew=0.5,
                lw=0.85, ms=2.0, zorder=3)


# ----------------------------- data loading ------------------------------ #
def load_states():
    if not os.path.isdir(DATA_SNAPSHOT):
        sys.exit("missing the data snapshot at\n  " + DATA_SNAPSHOT +
                 "\nrun this script from inside the NetRAD_Blind_Results "
                 "package")
    files = sorted(glob.glob(os.path.join(
        DATA_SNAPSHOT, "netrad_roc_*_v1.json")))
    states = {}
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                st = json.load(fh)
        except Exception:
            continue
        if st.get("complete"):
            ts = os.path.basename(f).split("_")[2]
            states[ts] = st
    # defensive: every kept arm must be complete in every checkpoint
    for ts, st in states.items():
        if "du_ccebm" not in st["h0"] or len(st["h0"]["du_ccebm"]) != 400:
            sys.exit("checkpoint %s lacks the du_ccebm arm; use the "
                     "package shipped with this release" % ts)
        for key, acc in st["levels"].items():
            if len(acc.get("du_ccebm", [])) != 80:
                sys.exit("checkpoint %s level %s lacks du_ccebm H1 stats"
                         % (ts, key))
    if len(states) != 21:
        sys.exit("expected exactly 21 complete checkpoints, got %d -- "
                 "refusing to reproduce on a partial snapshot"
                 % len(states))
    print("verified 21/21 complete checkpoints "
          "(%d files scanned, du_ccebm arm checked)" % len(files),
          flush=True)
    return states


def auc(h0, h1):
    """Mann-Whitney AUC, ties=0.5 (registered scheme)."""
    h0 = np.asarray([x for x in h0 if np.isfinite(x)], float)
    h1 = np.asarray([x for x in h1 if np.isfinite(x)], float)
    if not len(h0) or not len(h1):
        return float("nan")
    gt = (h1[:, None] > h0[None, :]).sum()
    eq = (h1[:, None] == h0[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(h0) * len(h1)))


# ------------------------------- figures --------------------------------- #
def mean_roc(states, key, arm):
    """Mean ROC over acquisitions at the empirical-P_fa grid."""
    pds = []
    for ts in sorted(states):
        st = states[ts]
        h0v = np.asarray(st["h0"][arm], float)
        h1v = np.asarray(st["levels"][key][arm], float)
        h0ok, h1ok = h0v[np.isfinite(h0v)], h1v[np.isfinite(h1v)]
        if len(h0ok) == 0 or len(h1ok) == 0:
            continue
        thrs = np.quantile(h0ok, 1 - PFA_GRID)
        pds.append([float(np.mean(h1ok > t)) for t in thrs])
    return np.mean(pds, axis=0) if pds else None


def power_pfa(states):
    # du_ccebm thresholds are NOT in st["thr"] (appended after the
    # registered run froze); calibrate them in-memory from its own 400
    # H0 statistics with the identical per-file scheme.
    states = {ts: dict(st) for ts, st in states.items()}
    for ts, st in states.items():
        h0v = np.asarray(st["h0"]["du_ccebm"], float)
        h0ok = h0v[np.isfinite(h0v)]
        thr = (float(np.quantile(h0ok, 1 - PFA_REG))
               if len(h0ok) else float("nan"))
        st["thr"] = dict(st.get("thr", {}))
        st["thr"]["du_ccebm"] = thr
    pw = {("%g" % s): {a: [] for a in ARMS} for s in SCR_LEVELS}
    per_file = []
    for scr in SCR_LEVELS:
        key = "%g" % scr
        for ts in sorted(states):
            st = states[ts]
            if key not in st["levels"]:
                continue
            for a in ARMS:
                thr = st["thr"].get(a, float("nan"))
                h1v = np.asarray(st["levels"][key][a], float)
                h1ok = h1v[np.isfinite(h1v)]
                pd = float(np.mean(h1ok > thr)) if len(h1ok) \
                    else float("nan")
                pw[key][a].append(pd)
                per_file.append((key, a, ts, pd))
    return pw, per_file


def fig_roc_norival_du(states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman",
                                        "STIXGeneral", "DejaVu Serif"],
                         "pdf.fonttype": 42})

    # per-SCR mean AUC + du_ccebm-significance vs every rival
    mean_auc = {}
    for scr in SCR_LEVELS:
        key = "%g" % scr
        mean_auc[key] = {}
        for a in ARMS:
            vals = [auc(states[ts]["h0"][a], states[ts]["levels"][key][a])
                    for ts in sorted(states) if key in states[ts]["levels"]]
            mean_auc[key][a] = float(np.nanmean(vals))
    # paired bootstrap over acquisitions (registered scheme)
    n = len(states)
    ts_sorted = sorted(states)
    rng = np.random.default_rng(SEED)
    boot = {k: {a: [] for a in ARMS} for k in mean_auc}
    per_scr_auc = {k: {a: [auc(states[ts]["h0"][a],
                               states[ts]["levels"][k][a])
                           for ts in ts_sorted if k in states[ts]["levels"]]
                       for a in ARMS} for k in mean_auc}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for key, rows in per_scr_auc.items():
            for a in ARMS:
                v = [rows[a][i] for i in idx if i < len(rows[a])]
                v = [x for x in v if np.isfinite(x)]
                boot[key][a].append(float(np.mean(v)) if v
                                    else float("nan"))
    sig = {}
    for key in mean_auc:
        for a in ARMS:
            if a == "du_ccebm":
                continue
            d = (np.asarray(boot[key]["du_ccebm"], float)
                 - np.asarray(boot[key][a], float))
            d = d[np.isfinite(d)]
            sig[(key, a)] = bool(float(np.percentile(d, 2.5)) > 0)

    letters = "abcde"
    fig = plt.figure(figsize=(7.16, 2.55))
    gs = fig.add_gridspec(1, len(SCR_LEVELS), left=0.075, right=0.995,
                          top=0.895, bottom=0.36, wspace=0.10)
    axes = [fig.add_subplot(gs[0, c]) for c in range(len(SCR_LEVELS))]
    handles = []
    for a in ARMS:
        handles.append(Line2D([], [], label=LEGEND[a], **_style_line(a)))
    curves_csv = []
    for c, scr in enumerate(SCR_LEVELS):
        ax = axes[c]
        key = "%g" % scr
        for a in ARMS:
            y = mean_roc(states, key, a)
            if y is None:
                continue
            kw = _style_line(a)
            ax.plot(PFA_GRID, y, markevery=3, **kw)
            curves_csv.append((key, a, y))
        ax.plot(PFA_GRID, PFA_GRID, color="0.6", lw=0.5, ls=":", zorder=1)
        ax.axvline(PFA_REG, color="0.45", lw=0.6, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(PFA_GRID[0] * 0.8, PFA_GRID[-1] * 1.25)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xticks([0.005, 0.05, 0.5])
        ax.grid(True, which="major", ls=":", c="0.6", lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(f"({letters[c]}) SCR ${scr:+g}$ dB", fontsize=7.5)
        ax.set_xlabel(r"$P_{\rm fa}$ (empirical)")
        if c == 0:
            ax.set_ylabel(r"mean $P_{\rm d}$")
            ax.text(PFA_REG * 1.3, 0.50, r"$P_{\rm fa}$ = 0.01",
                    rotation=90, fontsize=5.6, color="0.35",
                    ha="left", va="center")
        lines = [f"{SHORT[a]:<6}{mean_auc[key][a]:.2f}".replace(
            "0.", " .", 1) + ("*" if a != "du_ccebm"
                                      and sig.get((key, a)) else "")
                 for a in ARMS]
        ax.text(0.035, 0.965, "\n".join(lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=5.2, family="monospace",
                linespacing=1.22, zorder=6,
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec="0.75", lw=0.4, alpha=0.92))
    axes[-1].text(0.84, 0.47, "chance", fontsize=5.6, color="0.45",
                  rotation=19, ha="center", va="center",
                  transform=axes[-1].transAxes)
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.535, 0.018), ncol=5, frameon=False,
               fontsize=6.5, handlelength=2.2, columnspacing=1.1,
               handletextpad=0.45)
    fig.text(0.995, 0.002,
             "* DU-CCEBM significantly better than the rival / best blind "
             "arm (95% CI, paired bootstrap)", ha="right", fontsize=5.4,
             style="italic", color="0.25")
    fig.suptitle(f"Empirical ROC, mean over {len(states)} NetRAD "
                 f"acquisitions, offset +21 Hz, $M=128$, "
                 "400 H0 / 80 H1 windows each (Path A)\n"
                 "cERBM and DU-cERBM removed; DU-CCEBM is the proposed "
                 "deep arm (9 arms total)",
                 fontsize=7.5, y=1.06)
    os.makedirs(FIGS_OUT, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(
            FIGS_OUT, "figNETN1D_roc_norival_du.%s" % ext), **kw)
    plt.close(fig)
    print("  wrote figNETN1D_roc_norival_du.pdf/.png", flush=True)
    return curves_csv


def fig_power_norival_du(pw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman",
                                        "STIXGeneral", "DejaVu Serif"],
                         "pdf.fonttype": 42})

    n_files = len(next(iter(pw.values()))[ARMS[0]])
    fig = plt.figure(figsize=(7.16, 2.75))
    gs = fig.add_gridspec(1, 1, left=0.085, right=0.995, top=0.90,
                          bottom=0.30)
    ax = fig.add_subplot(gs[0, 0])
    handles = []
    x = np.asarray(SCR_LEVELS, float)
    rows = []
    for a in ARMS:
        m, lo, hi = [], [], []
        for scr in SCR_LEVELS:
            v = np.asarray(pw["%g" % scr][a], float)
            v = v[np.isfinite(v)]
            boot = np.random.default_rng(SEED + int(scr)).choice(
                v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
            m.append(v.mean())
            lo.append(np.percentile(boot, 2.5))
            hi.append(np.percentile(boot, 97.5))
            rows.append(["%g" % scr, a, len(v), f"{v.mean():.4f}",
                         f"{np.percentile(boot, 2.5):.4f}",
                         f"{np.percentile(boot, 97.5):.4f}"])
        m, lo, hi = map(np.asarray, (m, lo, hi))
        handles.append(Line2D([], [], label=LEGEND[a], **_style_line(a)))
        ax.errorbar(x, m, yerr=[m - lo, hi - m], capsize=0, elinewidth=0.5,
                    alpha=0.95, **_style_line(a))
    ax.set_xticks(x)
    ax.set_xlim(-27, -3)
    ax.set_ylim(0.0, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("SCR (dB)")
    ax.set_ylabel(r"$P_{\rm d}$ at $P_{\rm fa}=0.01$")
    ax.grid(True, which="major", ls=":", c="0.6", lw=0.4)
    ax.set_axisbelow(True)
    ax.set_title(f"Detection power vs SCR, registered operating point "
                 f"$P_{{\\rm fa}} = 0.01$ (per-file calibrated thresholds), "
                 f"mean $\\pm$ 95% CI over {n_files} acquisitions\n"
                 "cERBM and DU-cERBM removed; DU-CCEBM is the proposed "
                 "deep arm (9 arms total)", fontsize=7.5)
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.54, 0.02), ncol=5, frameon=False,
               fontsize=6.4, handlelength=2.2, columnspacing=1.0,
               handletextpad=0.5)
    os.makedirs(FIGS_OUT, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(
            FIGS_OUT, "figNETN2D_power_norival_du.%s" % ext), **kw)
    plt.close(fig)
    print("  wrote figNETN2D_power_norival_du.pdf/.png", flush=True)
    return rows


def fig_peracq_norival_du(states, key="-15"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"savefig.bbox": "tight",
                         "savefig.pad_inches": 0.02,
                         "font.family": "serif",
                         "font.serif": ["Times New Roman",
                                        "STIXGeneral", "DejaVu Serif"],
                         "pdf.fonttype": 42,
                         "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 8, "axes.linewidth": 0.6,
                         "grid.linewidth": 0.4,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    fig = plt.figure(figsize=(7.16, 2.45))
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.995, top=0.80,
                          bottom=0.30, wspace=0.14)
    arms = [("du_ccebm", "(a) DU-CCEBM (proposed)"),
            ("kelly_glrt", "(b) Kelly GLRT (best classical)")]
    for c, (arm, title) in enumerate(arms):
        ax = fig.add_subplot(gs[0, c])
        col = STYLE[arm][0]
        curves = []
        for ts in sorted(states):
            st = states[ts]
            h0v = np.asarray(st["h0"][arm], float)
            h1v = np.asarray(st["levels"][key][arm], float)
            h0ok, h1ok = h0v[np.isfinite(h0v)], h1v[np.isfinite(h1v)]
            thrs = np.quantile(h0ok, 1 - PFA_GRID)
            curves.append([float(np.mean(h1ok > t)) for t in thrs])
        for y in curves:
            ax.plot(PFA_GRID, y, color=col, lw=0.5, alpha=0.28, zorder=2)
        mu = np.mean(curves, axis=0)
        aucs = [auc(states[ts]["h0"][arm], states[ts]["levels"][key][arm])
                for ts in sorted(states)]
        ax.plot(PFA_GRID, mu, color=col, lw=1.5, marker=STYLE[arm][2],
                mfc="white", mew=0.6, ms=2.6, markevery=4, zorder=5,
                label=f"mean (AUC {np.mean(aucs):.3f})")
        ax.plot(PFA_GRID, PFA_GRID, color="0.6", lw=0.5, ls=":", zorder=1)
        ax.axvline(PFA_REG, color="0.45", lw=0.6, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(PFA_GRID[0] * 0.8, PFA_GRID[-1] * 1.25)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xticks([0.005, 0.05, 0.5])
        ax.grid(True, which="major", ls=":", c="0.6", lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(f"{title}, SCR ${float(key):+g}$ dB", fontsize=7.5)
        ax.set_xlabel(r"$P_{\rm fa}$ (empirical)")
        if c == 0:
            ax.set_ylabel(r"$P_{\rm d}$")
        ax.legend(loc="lower right", frameon=False, fontsize=6)
        ax.text(0.97, 0.90,
                f"{len(curves)} acquisitions\nthin: per-file ROC",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=5.8, color="0.35")
    fig.suptitle(f"Per-acquisition ROC spread at SCR {key} dB "
                 f"({len(states)} NetRAD acquisitions, Path A; "
                 "DU-CCEBM vs Kelly, no CERBM/DU-cERBM)", fontsize=7.5,
                 y=0.985)
    os.makedirs(FIGS_OUT, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(
            FIGS_OUT, "figNETN3D_roc_peracq_norival_du.%s" % ext), **kw)
    plt.close(fig)
    print("  wrote figNETN3D_roc_peracq_norival_du.pdf/.png", flush=True)


# ------------------------------- tables ---------------------------------- #
def write_curves_csv(rows):
    path = os.path.join(DATA_OUT, "netrad_roc_curves_norival_du.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scr_db", "arm", "pfa_empirical", "pd_mean_over_files"])
        for key, a, y in rows:
            for p, pd in zip(PFA_GRID, y):
                w.writerow([key, a, f"{p:.6g}", f"{pd:.6f}"])
    print("  wrote", path, flush=True)


def write_auc_csv(states):
    path = os.path.join(DATA_OUT, "netrad_auc_norival_du.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scr_db", "arm", "n_files", "auc_mean", "auc_sd"])
        for scr in SCR_LEVELS:
            key = "%g" % scr
            for a in ARMS:
                per = [auc(states[ts]["h0"][a], states[ts]["levels"][key][a])
                       for ts in sorted(states)
                       if key in states[ts]["levels"]]
                v = np.asarray(per, float)
                w.writerow([key, a, len(v), f"{np.nanmean(v):.4f}",
                            f"{np.nanstd(v, ddof=1):.4f}"])
    print("  wrote", path, flush=True)


def write_effects_csv(states):
    """du_ccebm vs each rival: paired bootstrap d-AUC per SCR."""
    path = os.path.join(DATA_OUT, "netrad_auc_effects_norival_du.csv")
    ts_sorted = sorted(states)
    per_scr = {}
    for scr in SCR_LEVELS:
        key = "%g" % scr
        rows = {a: [auc(states[ts]["h0"][a], states[ts]["levels"][key][a])
                    for ts in ts_sorted if key in states[ts]["levels"]]
                for a in ARMS}
        per_scr[key] = rows
    rng = np.random.default_rng(SEED)
    n = len(ts_sorted)
    boot = {k: {a: [] for a in ARMS} for k in per_scr}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for key, rows in per_scr.items():
            for a in ARMS:
                v = [rows[a][i] for i in idx if i < len(rows[a])]
                v = [x for x in v if np.isfinite(x)]
                boot[key][a].append(float(np.mean(v)) if v
                                    else float("nan"))
    comps = []
    for key in sorted(per_scr, key=float):
        for a in ARMS:
            if a == "du_ccebm":
                continue
            d = (np.asarray(boot[key]["du_ccebm"], float)
                 - np.asarray(boot[key][a], float))
            d = d[np.isfinite(d)]
            p = float((d <= 0).mean())
            lo, hi = np.percentile(d, [2.5, 97.5])
            comps.append([key, a, float(d.mean()), float(lo), float(hi),
                          p, lo > 0])
    # Holm step-down across the DU-CCEBM-vs-rival family
    order = sorted(range(len(comps)), key=lambda i: comps[i][5])
    m = len(order)
    holm = [False] * m
    for rank, i in enumerate(order):
        if comps[i][5] <= 0.05 / (m - rank):
            holm[rank] = True
        else:
            break
    for rank, i in enumerate(order):
        comps[i].append(bool(holm[rank]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scr_db", "rival", "mean_d_auc", "ci_lo", "ci_hi",
                    "p_one_sided_bootstrap", "sig_95ci_positive",
                    "sig_holm_stepdown"])
        for r in comps:
            w.writerow([r[0], r[1], f"{r[2]:+.4f}", f"{r[3]:+.4f}",
                        f"{r[4]:+.4f}", f"{r[5]:.4f}", r[6], r[7]])
    print("  wrote", path, flush=True)
    return comps


def write_power_csvs(pw, per_file):
    path = os.path.join(DATA_OUT, "netrad_power_pfa01_norival_du.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scr_db", "arm", "n_files", "pd_mean", "ci_lo", "ci_hi"])
        for scr in SCR_LEVELS:
            key = "%g" % scr
            for a in ARMS:
                v = np.asarray(pw[key][a], float)
                v = v[np.isfinite(v)]
                boot = np.random.default_rng(SEED + int(scr)).choice(
                    v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
                w.writerow([key, a, len(v), f"{v.mean():.4f}",
                            f"{np.percentile(boot, 2.5):.4f}",
                            f"{np.percentile(boot, 97.5):.4f}"])
    print("  wrote", path, flush=True)
    path = os.path.join(DATA_OUT, "netrad_power_peracq_norival_du.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scr_db", "arm", "timestamp", "pd"])
        for key, a, ts, pd in per_file:
            w.writerow([key, a, ts, f"{pd:.4f}"])
    print("  wrote", path, flush=True)


def main():
    print("== repro_norival_du: NetRAD ROC & power, 9 arms "
          "(rivals + DU-CCEBM; NO cERBM/DU-cERBM) ==", flush=True)
    states = load_states()
    print("acquisitions:\n" + "\n".join(sorted(states)), flush=True)

    os.makedirs(DATA_OUT, exist_ok=True)
    pw, per_file = power_pfa(states)
    print("\n-- data tables", flush=True)
    write_auc_csv(states)
    write_effects_csv(states)
    write_power_csvs(pw, per_file)
    print("\n-- figures", flush=True)
    curves_rows = fig_roc_norival_du(states)
    write_curves_csv(curves_rows)
    fig_power_norival_du(pw)
    fig_peracq_norival_du(states)
    print("\nDONE. outputs in figs_norival_DU/ + data_norival_DU/",
          flush=True)


if __name__ == "__main__":
    main()
