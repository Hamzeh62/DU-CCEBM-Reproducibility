"""
Real-IPIX-data driver, Part C: MMD domain-alignment (Stage 4 of
FABLE_SUGGESTIONS_IMPLEMENTATION_PROMPT.md), now runnable because real
.cdf files were uploaded mid-session.

Design (exactly matching the paper's Sec. 2.2 motivation): train
DU-CCEBM primarily on SYNTHETIC data (same ablation-suite scale as
Stage 1/3: T=256, L_net=8, L_filt=8, K_white=5, seed=42), with an
OPTIONAL auxiliary MMD term that pulls phi(synthetic batch) toward
phi(real batch) during training. "Real batch" is built ONLY from
genuinely-recorded IPIX windows (bypassing _inject_targets_dualpol,
exactly as in run_real_B2_leak_fix.py) whose dataset_full index is in
the TRAIN-eligible portion (i.e. NOT in the same held-out test set used
for final evaluation -- reusing the exact index bookkeeping validated in
run_real_B2_leak_fix.py).

Two models are trained and compared:
  "Full (synthetic only)"      -- lam_mmd=0.0 (baseline, no real data touched)
  "Full (synthetic) + MMD(real)" -- lam_mmd=0.3, real_feature_dataset=
                                     genuine-real TRAIN-eligible windows

Both are evaluated on the SAME leak-free real held-out test set (genuine
windows in TEST split only, per run_real_B2_leak_fix.py's index logic)
for a fair, apples-to-apples comparison of real-data generalisation.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '.')
import du_ccebm_simulation as sim

OUT = Path("./revision_results")
OUT.mkdir(exist_ok=True)
MASTER = OUT / "REVISION_REAL_RESULTS.json"
master = json.loads(MASTER.read_text()) if MASTER.exists() else {}


def save_master(d):
    with open(MASTER, "w") as f:
        json.dump(d, f, indent=2, default=str)
    import shutil
    shutil.copy(MASTER, "/mnt/user-data/outputs/REVISION_REAL_RESULTS.json")


STAGE_KEY = "real_C_mmd_domain_alignment"
if STAGE_KEY in master and master[STAGE_KEY].get("done"):
    print(f"[skip] {STAGE_KEY} already completed.")
    sys.exit(0)

print("=" * 70)
print("REAL-C: MMD domain alignment (synthetic-trained DU-CCEBM + real phi)")
print("=" * 70)

# ── Reproduce the real-data index bookkeeping from real_A / real_B2 ────
a = master["real_A_train_du_ccebm"]
cfg_real = dict(sim.CFG)
cfg_real.update(a["config"])
cfg_real.update({"ipix_dir": "ipix_data", "device": "cpu"})

sim.set_seed(cfg_real["seed"])
dataset_full = sim.IPIXDataset(cfg_real, mode="train")
n_full = len(dataset_full)
assert n_full == a["n_samples_loaded_total"]

np.random.seed(42)
idx = np.random.choice(n_full, size=a["n_samples_used_subset"], replace=False)
n = len(idx)
n_test = int(n * cfg_real["test_frac"])
n_val = int(n * cfg_real["val_frac"])
n_train = n - n_val - n_test
gen = torch.Generator().manual_seed(cfg_real["seed"])
perm = torch.randperm(n, generator=gen).tolist()
train_pos, val_pos, test_pos = perm[:n_train], perm[n_train:n_train+n_val], perm[n_train+n_val:]
test_dataset_full_indices = set(idx[test_pos].tolist())
train_val_dataset_full_indices = set(idx[train_pos].tolist()) | set(idx[val_pos].tolist())

orig_inject = dataset_full._inject_targets_dualpol
dataset_full._inject_targets_dualpol = lambda clutter_pairs: ([], [])
files = sorted(Path(cfg_real["ipix_dir"]).glob("*.cdf"))
genuine_samples, genuine_labels = dataset_full._load_ipix(files)
dataset_full._inject_targets_dualpol = orig_inject
genuine_labels = np.asarray(genuine_labels, dtype=np.float32)
n_genuine = len(genuine_labels)
genuine_hh_all = np.stack([s[0] for s in genuine_samples])
genuine_vv_all = np.stack([s[1] for s in genuine_samples])

# TEST set for final evaluation: genuine windows in test_pos ONLY (a strict
# subset of what real_B2 used -- there it was "all genuine minus train/val",
# here it's specifically "genuine AND in the test split" for the tightest
# possible train/eval separation, since MMD in this script gets to see
# train+val genuine windows during training and must not see test ones).
eval_mask = np.array([i in test_dataset_full_indices for i in range(n_genuine)])
mmd_train_mask = np.array([(i not in test_dataset_full_indices) for i in range(n_genuine)])
print(f"[data] genuine pool n={n_genuine}: "
      f"{eval_mask.sum()} reserved for evaluation (test split), "
      f"{mmd_train_mask.sum()} available for MMD training (train+val+untouched)")

eval_hh, eval_vv, eval_labels = genuine_hh_all[eval_mask], genuine_vv_all[eval_mask], genuine_labels[eval_mask]
print(f"[data] real held-out eval set: n={len(eval_labels)} "
      f"(target={int(eval_labels.sum())}, clutter={int(len(eval_labels)-eval_labels.sum())})")

mmd_hh, mmd_vv = genuine_hh_all[mmd_train_mask], genuine_vv_all[mmd_train_mask]
mmd_x = np.zeros((len(mmd_hh), 2, 2, cfg_real["T"]), dtype=np.float32)
mmd_x[:, 0, 0] = mmd_hh.real; mmd_x[:, 1, 0] = mmd_hh.imag
mmd_x[:, 0, 1] = mmd_vv.real; mmd_x[:, 1, 1] = mmd_vv.imag
real_feature_dataset = torch.utils.data.TensorDataset(torch.from_numpy(mmd_x))
print(f"[data] MMD real_feature_dataset: n={len(real_feature_dataset)} "
      f"(T={cfg_real['T']}, real windows for phi_real only, no labels used)")

# ── Synthetic training data (matched to Stage 1/3 ablation-suite scale) ──
cfg_syn = dict(sim.CFG)
cfg_syn.update({
    "ipix_dir": "./no_such_dir_forces_synthetic",
    "T": 256, "L_net": 8, "L_filt": 8, "K_white": 5,
    "batch": 128, "lr": 1e-3,
    "epochs": 20, "patience": 8,
    "val_frac": 0.15, "test_frac": 0.10,
    "pfa_target": 1e-3,
    "lam1": 1.0, "lam2": 0.5, "lam3": 0.1,
    "seed": 42,
})
sim.set_seed(cfg_syn["seed"])
ds_syn = sim.IPIXDataset.__new__(sim.IPIXDataset)
ds_syn.cfg, ds_syn.mode, ds_syn.T = cfg_syn, "train", cfg_syn["T"]
ds_syn.pols, ds_syn.P_ref = cfg_syn["pols"], cfg_syn["P_ref"]
ds_syn.samples, ds_syn.labels = ds_syn._synthetic_dataset(n_clutter=1200, n_target=1200)
print(f"[data] synthetic training pool: n={len(ds_syn)}")


def eval_on_real(model):
    x = np.zeros((len(eval_labels), 2, 2, cfg_real["T"]), dtype=np.float32)
    x[:, 0, 0] = eval_hh.real; x[:, 1, 0] = eval_hh.imag
    x[:, 0, 1] = eval_vv.real; x[:, 1, 1] = eval_vv.imag
    model.eval()
    with torch.no_grad():
        scores = []
        B = 256
        for s in range(0, len(eval_labels), B):
            sc, *_ = model(torch.from_numpy(x[s:s+B]))
            scores.append(sc.numpy())
        scores = np.concatenate(scores)
    pt, lo, hi, _, _ = sim.bootstrap_auc_ci(scores, eval_labels, n_boot=1000, seed=42)
    min_pfa = sim.min_representable_pfa(eval_labels)
    pfa_report = 1e-2 if min_pfa <= 1e-2 else round(min_pfa, 4)
    pt_pd, lo_pd, hi_pd, _, _ = sim.bootstrap_pd_ci(scores, eval_labels, pfa=pfa_report, n_boot=1000, seed=42)
    return scores, {"auc": {"point": pt, "ci95": [lo, hi]},
                     "pd": {"point": pt_pd, "ci95": [lo_pd, hi_pd]},
                     "reported_pfa_operating_point": pfa_report,
                     "min_representable_nonzero_pfa": min_pfa}


results = {}

# ── (1) Full (synthetic only), lam_mmd=0.0 ──────────────────────────────
cfg_a = dict(cfg_syn); cfg_a.update({"save_dir": str(OUT / "real_C_no_mmd"), "lam_mmd": 0.0})
ckpt_a_path = Path(cfg_a["save_dir"]) / "best_model.pt"
if ckpt_a_path.exists():
    print(f"\n[resume] found existing checkpoint {ckpt_a_path}, skipping retraining "
          f"of 'Full (synthetic only)'.")
    ck = torch.load(ckpt_a_path, map_location="cpu", weights_only=False)
    model_no_mmd = sim.DUCCEBM(L_net=cfg_syn["L_net"], N=cfg_syn["N"],
                                L_filt=cfg_syn["L_filt"], K_white=cfg_syn["K_white"])
    model_no_mmd.load_state_dict(ck["model"])
    hist_no_mmd = ck["history"]
    wall_no_mmd = float(json.loads(
        (OUT / "real_C_no_mmd_progress.json").read_text())["wall_clock_s"])
else:
    print("\n--- Training: Full (synthetic only, lam_mmd=0.0) ---")
    t0 = time.time()
    model_no_mmd, hist_no_mmd, *_ = sim.train(
        cfg_a, ds_syn, model_kwargs={}, verbose_progress=False, early_stop_metric="auc_val",
        progress_path=str(OUT / "real_C_no_mmd_progress.json"))
    wall_no_mmd = time.time() - t0
scores_no_mmd, eval_no_mmd = eval_on_real(model_no_mmd)
print(f"  wall={wall_no_mmd:.1f}s epochs={len(hist_no_mmd['loss'])} "
      f"real-eval AUC={eval_no_mmd['auc']['point']:.4f} "
      f"[{eval_no_mmd['auc']['ci95'][0]:.4f},{eval_no_mmd['auc']['ci95'][1]:.4f}]")

# ── (2) Full (synthetic) + MMD(real), lam_mmd=0.3 ───────────────────────
print("\n--- Training: Full (synthetic) + MMD(real), lam_mmd=0.3 ---")
cfg_b = dict(cfg_syn); cfg_b.update({"save_dir": str(OUT / "real_C_with_mmd"),
                                      "lam_mmd": 0.3, "mmd_real_batch": 32})
t0 = time.time()
model_mmd, hist_mmd, *_ = sim.train(
    cfg_b, ds_syn, model_kwargs={}, verbose_progress=False, early_stop_metric="auc_val",
    real_feature_dataset=real_feature_dataset,
    progress_path=str(OUT / "real_C_with_mmd_progress.json"))
wall_mmd = time.time() - t0
scores_mmd, eval_mmd = eval_on_real(model_mmd)
print(f"  wall={wall_mmd:.1f}s epochs={len(hist_mmd['loss'])} "
      f"real-eval AUC={eval_mmd['auc']['point']:.4f} "
      f"[{eval_mmd['auc']['ci95'][0]:.4f},{eval_mmd['auc']['ci95'][1]:.4f}]")

# ── paired significance test (same real eval set, same labels) ─────────
pt, lo, hi, sig = sim.bootstrap_auc_difference_ci(scores_mmd, scores_no_mmd, eval_labels, n_boot=1000, seed=7)
print(f"\n[compare] MMD vs no-MMD on real held-out eval: "
      f"delta_AUC={pt:+.4f} [{lo:+.4f},{hi:+.4f}] significant={sig}")

training_diverged = any(not np.isfinite(v) for v in hist_mmd["loss"])

stage = {
    "done": True,
    "n_eval_real_test": int(len(eval_labels)),
    "n_eval_real_target": int(eval_labels.sum()),
    "n_mmd_real_train_pool": int(len(real_feature_dataset)),
    "training_diverged_or_nonfinite": bool(training_diverged),
    "no_mmd": {
        "wall_clock_s": wall_no_mmd, "epochs_run": len(hist_no_mmd["loss"]),
        "loss_history": hist_no_mmd["loss"], "mmd_history": hist_no_mmd["mmd"],
        "real_eval": eval_no_mmd,
    },
    "with_mmd": {
        "wall_clock_s": wall_mmd, "epochs_run": len(hist_mmd["loss"]),
        "loss_history": hist_mmd["loss"], "mmd_history": hist_mmd["mmd"],
        "real_eval": eval_mmd,
        "lam_mmd": cfg_b["lam_mmd"], "mmd_real_batch": cfg_b["mmd_real_batch"],
    },
    "auc_diff_with_mmd_minus_no_mmd_on_real_eval": {
        "point": pt, "ci95": [lo, hi], "statistically_significant_at_0.05": bool(sig),
    },
}
master[STAGE_KEY] = stage
save_master(master)
print("\n[REAL-C SUMMARY]")
print(json.dumps({k: v for k, v in stage.items()
                   if not (isinstance(v, dict) and any('history' in kk for kk in v))},
                  indent=2, default=str))
