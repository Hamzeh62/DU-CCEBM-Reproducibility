"""
Real-IPIX-data driver, Part A: train DU-CCEBM on real McMaster IPIX data.

Uses the SAME architecture/hyperparameters as the prior session's
run_real_ipix_experiment_v2.py (T=128, overlap=64, max_windows_per_file=150,
L_net=6, L_filt=6, K_white=4, batch=16, epochs=15, patience=8, seed=42)
for direct comparability with MASTER_REAL_RESULTS_original_session.json,
but now uses ALL 14 uploaded files (the prior session only had 4) and a
larger subsample (2400 vs. 1200) to reduce the "n_target=45 is small,
high variance" caveat that session's own results explicitly flagged.

No checkpoint from a real-data training run survived from the prior
session (only the JSON summary did -- confirmed by exhaustive search of
this session's uploaded zip), so this training run is necessary before
ANMF/MMD comparisons against DU-CCEBM on real data are possible.
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


def save_master(d):
    with open(MASTER, "w") as f:
        json.dump(d, f, indent=2, default=str)
    try:
        import shutil
        shutil.copy(MASTER, "/mnt/user-data/outputs/REVISION_REAL_RESULTS.json")
    except Exception as e:
        print(f"[warn] could not mirror to outputs: {e}")


master = {}
if MASTER.exists():
    master = json.loads(MASTER.read_text())

STAGE_KEY = "real_A_train_du_ccebm"
if STAGE_KEY in master and master[STAGE_KEY].get("done"):
    print(f"[skip] {STAGE_KEY} already completed.")
else:
    t_start = time.time()
    sim.set_seed(42)

    cfg = dict(sim.CFG)
    cfg.update({
        "ipix_dir": "ipix_data",
        "T": 128, "overlap": 64, "max_windows_per_file": 150,
        "L_net": 6, "L_filt": 6, "K_white": 4,
        "epochs": 15, "batch": 16, "patience": 8,
        "save_dir": str(OUT / "real_du_ccebm"),
        "device": "cpu", "seed": 42,
    })
    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REAL-A: Training DU-CCEBM on ALL 14 uploaded real IPIX files")
    print(f"  T={cfg['T']} L_net={cfg['L_net']} epochs={cfg['epochs']} "
          f"batch={cfg['batch']} patience={cfg['patience']}")
    print("=" * 70)

    dataset_full = sim.IPIXDataset(cfg, mode="train")
    n_target_total = int(dataset_full.labels.sum())
    n_clutter_total = int(len(dataset_full.labels) - dataset_full.labels.sum())
    print(f"[data] loaded {len(dataset_full)} total windows from 14 real files "
          f"(target={n_target_total}, clutter={n_clutter_total})")

    np.random.seed(42)
    N_USE = min(2400, len(dataset_full))
    idx = np.random.choice(len(dataset_full), size=N_USE, replace=False)
    dataset = torch.utils.data.Subset(dataset_full, idx)
    print(f"[data] using subsample of {len(dataset)} / {len(dataset_full)}")

    t0 = time.time()
    model, history, test_loader, device = sim.train(
        cfg, dataset, model_kwargs={}, verbose_progress=False,
        early_stop_metric="pd_val",   # matches prior session's stopper.step(pd_val, ...)
        progress_path=str(OUT / "real_A_progress.json"))
    wall_s = time.time() - t0
    print(f"\n[train] done in {wall_s:.1f}s, {len(history['loss'])} epochs run")

    stage = {
        "done": True,
        "files_used": sorted(p.name for p in Path(cfg["ipix_dir"]).glob("*.cdf")),
        "config": {k: cfg[k] for k in
                   ["T", "overlap", "max_windows_per_file", "L_net", "L_filt",
                    "K_white", "epochs", "batch", "patience", "seed"]},
        "n_samples_loaded_total": len(dataset_full),
        "n_target_loaded_total": n_target_total,
        "n_clutter_loaded_total": n_clutter_total,
        "n_samples_used_subset": len(dataset),
        "epochs_run": len(history["loss"]),
        "wall_clock_s": wall_s,
        "loss_history": history["loss"],
        "pd_val_history": history["pd_val"],
        "auc_val_history": history["auc_val"],
        "checkpoint_path": str(Path(cfg["save_dir"]) / "best_model.pt"),
    }
    master[STAGE_KEY] = stage
    save_master(master)

    print("\n[REAL-A SUMMARY]")
    print(json.dumps({k: v for k, v in stage.items()
                       if not k.endswith("_history")}, indent=2))
    print(f"\n[Run] Total wall-clock: {time.time()-t_start:.1f}s")
