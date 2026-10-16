"""
Real-IPIX-data driver, Part B2: FIX for train/val leakage found in
real_B_anmf_evaluation's "genuine_only_full_pool" result.

The original real_B run evaluated DU-CCEBM on ALL 23550 genuinely-
recorded windows, but the model was TRAINED on a 2400-sample subsample
of the full 45000-window pool (genuine + synthetic-injected), of which
1253 happened to be genuine windows -- so that evaluation included
~1253 samples the model's gradient (and, more weakly, its early-
stopping model-selection criterion) had already seen. This script
recomputes DU-CCEBM's genuine-only performance on a properly held-out
subset: every genuine window EXCEPT those whose underlying dataset_full
index appeared in the train OR validation split used by real_A_train.py
(test-split genuine windows, plus the ~41150 genuine+synthetic windows
never touched by the 2400-subsample at all, are legitimately unseen).

ANMF and CA-CFAR are recomputed on the SAME held-out set for a fair,
apples-to-apples comparison (they were not "trained" in the first place,
but recomputing on the identical index set keeps every number in this
table describing literally the same population of test windows).
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '.')
import du_ccebm_simulation as sim

OUT = Path("./revision_results")
MASTER = OUT / "REVISION_REAL_RESULTS.json"
master = json.loads(MASTER.read_text())


def save_master(d):
    with open(MASTER, "w") as f:
        json.dump(d, f, indent=2, default=str)
    import shutil
    shutil.copy(MASTER, "/mnt/user-data/outputs/REVISION_REAL_RESULTS.json")


a = master["real_A_train_du_ccebm"]
cfg = dict(sim.CFG)
cfg.update(a["config"])
cfg.update({"ipix_dir": "ipix_data", "device": "cpu",
            "save_dir": str(OUT / "real_du_ccebm")})

print("=" * 70)
print("REAL-B2: leak-free genuine-only evaluation (excludes train+val indices)")
print("=" * 70)

# ── Reproduce the EXACT indices used for training (same seeds as real_A) ──
sim.set_seed(cfg["seed"])
dataset_full = sim.IPIXDataset(cfg, mode="train")
n_full = len(dataset_full)
assert n_full == a["n_samples_loaded_total"], "dataset_full size mismatch -- not reproducible!"

np.random.seed(42)
idx = np.random.choice(n_full, size=a["n_samples_used_subset"], replace=False)  # (2400,)

n = len(idx)
n_test = int(n * cfg["test_frac"])
n_val = int(n * cfg["val_frac"])
n_train = n - n_val - n_test
gen = torch.Generator().manual_seed(cfg["seed"])
perm = torch.randperm(n, generator=gen).tolist()   # torch.utils.data.random_split's actual mechanism
train_pos, val_pos, test_pos = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]

train_val_dataset_full_indices = set(idx[train_pos].tolist()) | set(idx[val_pos].tolist())
test_dataset_full_indices = set(idx[test_pos].tolist())
print(f"[check] train+val touched {len(train_val_dataset_full_indices)} unique "
      f"dataset_full indices, test touched {len(test_dataset_full_indices)} "
      f"(should sum to {len(set(idx.tolist()))} unique of {len(idx)} total, "
      f"duplicates possible only if random_split overlapped, which it should not)")

# ── Genuine-only pool (bypass synthetic injection, exactly as in real_B) ──
orig_inject = dataset_full._inject_targets_dualpol
dataset_full._inject_targets_dualpol = lambda clutter_pairs: ([], [])
files = sorted(Path(cfg["ipix_dir"]).glob("*.cdf"))
genuine_samples, genuine_labels = dataset_full._load_ipix(files)
dataset_full._inject_targets_dualpol = orig_inject
genuine_labels = np.asarray(genuine_labels, dtype=np.float32)
n_genuine = len(genuine_labels)
print(f"[data] genuine-only pool: n={n_genuine} "
      f"(target={int(genuine_labels.sum())}, clutter={int(n_genuine - genuine_labels.sum())})")

# genuine_samples[:n_genuine] correspond EXACTLY to dataset_full indices
# [0, n_genuine) by construction (see run_real_B_anmf.py docstring / the
# _load_ipix ordering guarantee: genuine samples are appended first, in
# file/window order, before any synthetic injection).
held_out_mask = np.array([i not in train_val_dataset_full_indices
                           for i in range(n_genuine)])
n_leaked = n_genuine - held_out_mask.sum()
print(f"[fix] excluding {n_leaked} genuine windows whose dataset_full index "
      f"was used in train/val ({held_out_mask.sum()} genuine windows remain, "
      f"properly held out)")

genuine_hh_all = np.stack([s[0] for s in genuine_samples])
genuine_vv_all = np.stack([s[1] for s in genuine_samples])
hh = genuine_hh_all[held_out_mask]
vv = genuine_vv_all[held_out_mask]
labels = genuine_labels[held_out_mask]
print(f"[held-out genuine] n={len(labels)} (target={int(labels.sum())}, "
      f"clutter={int(len(labels)-labels.sum())})")

in_original_test = np.array([i in test_dataset_full_indices
                              for i in np.where(held_out_mask)[0]])
print(f"[cross-check] of these, {in_original_test.sum()} were also in the "
      f"original 240-sample test split; the remaining "
      f"{len(labels)-in_original_test.sum()} were simply never sampled into "
      f"the 2400-window training subsample at all (still legitimately unseen).")

ckpt = torch.load(a["checkpoint_path"], map_location="cpu", weights_only=False)
model = sim.DUCCEBM(L_net=cfg["L_net"], N=cfg["N"], L_filt=cfg["L_filt"], K_white=cfg["K_white"])
model.load_state_dict(ckpt["model"])
model.eval()

x = np.zeros((len(labels), 2, 2, cfg["T"]), dtype=np.float32)
x[:, 0, 0] = hh.real; x[:, 1, 0] = hh.imag
x[:, 0, 1] = vv.real; x[:, 1, 1] = vv.imag
with torch.no_grad():
    scores_du = []
    B = 512
    for s in range(0, len(labels), B):
        sc, *_ = model(torch.from_numpy(x[s:s+B]))
        scores_du.append(sc.numpy())
    scores_du = np.concatenate(scores_du)

clutter_idx = np.where(labels == 0)[0]
clutter_pool = [np.stack([hh[i], vv[i]]) for i in clutter_idx]
n_calib = min(200, len(clutter_pool))
R_hat, R_inv = sim.estimate_polarimetric_covariance(clutter_pool, n_calib=n_calib, diag_loading=1e-2, seed=42)
rho = R_hat[0, 1].real / np.sqrt(R_hat[0, 0].real * R_hat[1, 1].real)
X = np.stack([hh, vv], axis=1)
scores_anmf = sim.anmf_detect_batch(X, R_inv)

amp = np.abs(hh)
scores_cfar = amp.mean(-1)
scores_cfar = (scores_cfar - scores_cfar.min()) / (scores_cfar.max() - scores_cfar.min() + 1e-10)

min_pfa = sim.min_representable_pfa(labels)
pfa_report = 1e-2 if min_pfa <= 1e-2 else round(min_pfa, 4)


def auc_ci(s):
    pt, lo, hi, _, nf = sim.bootstrap_auc_ci(s, labels, n_boot=1000, seed=42)
    return {"point": pt, "ci95": [lo, hi], "n_boot_failed": int(nf)}


def pd_ci(s):
    pt, lo, hi, _, nf = sim.bootstrap_pd_ci(s, labels, pfa=pfa_report, n_boot=1000, seed=42)
    return {"point": pt, "ci95": [lo, hi], "n_boot_failed": int(nf)}


result = {
    "n": int(len(labels)), "n_target": int(labels.sum()), "n_clutter": int(len(labels)-labels.sum()),
    "n_excluded_as_train_val_leakage": int(n_leaked),
    "n_calib_clutter_for_R_hat": int(n_calib),
    "hh_vv_correlation_R_hat": float(rho),
    "reported_pfa_operating_point": pfa_report,
    "min_representable_nonzero_pfa": min_pfa,
    "DU-CCEBM": {"auc": auc_ci(scores_du), "pd": pd_ci(scores_du)},
    "ANMF":     {"auc": auc_ci(scores_anmf), "pd": pd_ci(scores_anmf)},
    "CA-CFAR":  {"auc": auc_ci(scores_cfar), "pd": pd_ci(scores_cfar)},
}
pt, lo, hi, sig = sim.bootstrap_auc_difference_ci(scores_du, scores_anmf, labels, n_boot=1000, seed=7)
result["auc_diff_DU_CCEBM_minus_ANMF"] = {
    "point": pt, "ci95": [lo, hi], "statistically_significant_at_0.05": bool(sig)}

print(f"\nDU-CCEBM AUC={result['DU-CCEBM']['auc']['point']:.4f} "
      f"[{result['DU-CCEBM']['auc']['ci95'][0]:.4f},{result['DU-CCEBM']['auc']['ci95'][1]:.4f}]")
print(f"ANMF     AUC={result['ANMF']['auc']['point']:.4f} "
      f"[{result['ANMF']['auc']['ci95'][0]:.4f},{result['ANMF']['auc']['ci95'][1]:.4f}]")
print(f"CA-CFAR  AUC={result['CA-CFAR']['auc']['point']:.4f} "
      f"[{result['CA-CFAR']['auc']['ci95'][0]:.4f},{result['CA-CFAR']['auc']['ci95'][1]:.4f}]")
print(f"DU-CCEBM vs ANMF: delta_AUC={pt:+.4f} [{lo:+.4f},{hi:+.4f}] significant={sig}")

master["real_B2_genuine_only_LEAK_FREE"] = {
    "done": True,
    "note": ("Corrects a train/val leakage bug in real_B_anmf_evaluation's "
             "'genuine_only_full_pool' result: that evaluation used ALL "
             "23550 genuine windows, of which 1253 had also been part of "
             "the 2400-window training subsample (used for gradient "
             "updates and/or early-stopping model selection). This result "
             "excludes every genuine window whose dataset_full index was "
             "used in train OR validation, keeping only genuinely held-out "
             "windows (test-split genuine windows, plus genuine windows "
             "never touched by the 2400-window subsample at all)."),
    "result": result,
}
save_master(master)
print("\n[REAL-B2 saved]")
