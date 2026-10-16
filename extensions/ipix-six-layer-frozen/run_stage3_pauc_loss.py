"""
Stage 3 driver (FABLE_SUGGESTIONS_IMPLEMENTATION_PROMPT.md):
  partial-AUC loss ablation, using the batch-size/alpha-mismatch-fixed
  HardNegativeBuffer + partial_auc_loss_buffered (Stage 0-verified core
  math, Stage-3-added buffering fix -- see FABLE_VERIFICATION_REPORT.md
  Sec. 2.1), wired into train_epoch()/train() behind cfg["lam_pauc"]
  (default 0.0 = off, so every prior result is unaffected).

Runs ONE new variant at the EXACT SAME data/splits/training budget as
the existing matched-scale ablation suite (T=256, L_net=8, L_filt=8,
K_white=5, batch=128, epochs=20, patience=8, N=2400 synthetic samples,
seed=42), directly comparable to the cached "Full DU-CCEBM" row:

  "Full + partial-AUC loss"  -- lam_pauc=0.5, pauc_alpha=1e-3,
                                 pauc_buffer_capacity=2000 (all other
                                 hyperparameters identical to "Full
                                 DU-CCEBM").

Per the implementation plan: "اگر بهبودی دیده نشد یا ناپایداری آموزش رخ
داد... این را دقیقاً همان‌طور گزارش کن" -- report whatever happens,
positive, negative, or null, without spin.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '.')
import du_ccebm_simulation as sim

OUT = Path("./revision_results")
OUT.mkdir(exist_ok=True)
MASTER = OUT / "REVISION_SYNTHETIC_RESULTS.json"


def save_master(d):
    with open(MASTER, "w") as f:
        json.dump(d, f, indent=2)
    try:
        import shutil
        shutil.copy(MASTER, "/mnt/user-data/outputs/REVISION_SYNTHETIC_RESULTS.json")
    except Exception as e:
        print(f"[warn] could not mirror to outputs: {e}")


master = {}
if MASTER.exists():
    master = json.loads(MASTER.read_text())

STAGE_KEY = "stage5_partial_auc_loss_ablation"
if STAGE_KEY in master and master[STAGE_KEY].get("done"):
    print(f"[skip] {STAGE_KEY} already completed.")
else:
    print("=" * 70)
    print("STAGE 3 (impl. plan) / stage5_partial_auc_loss_ablation")
    print("=" * 70)

    cfg_base = dict(sim.CFG)
    cfg_base.update({
        "ipix_dir": "./no_such_dir_forces_synthetic",
        "T": 256, "L_net": 8, "L_filt": 8, "K_white": 5,
        "batch": 128, "lr": 1e-3,
        "epochs": 20, "patience": 8,
        "val_frac": 0.15, "test_frac": 0.10,
        "pfa_target": 1e-3,
        "lam1": 1.0, "lam2": 0.5, "lam3": 0.1,
        "seed": 42,
    })
    N_CLUTTER, N_TARGET = 1200, 1200

    sim.set_seed(cfg_base["seed"])
    ds = sim.IPIXDataset.__new__(sim.IPIXDataset)
    ds.cfg, ds.mode, ds.T = cfg_base, "train", cfg_base["T"]
    ds.pols, ds.P_ref = cfg_base["pols"], cfg_base["P_ref"]
    ds.samples, ds.labels = ds._synthetic_dataset(n_clutter=N_CLUTTER, n_target=N_TARGET)
    print(f"[data] {len(ds)} samples "
          f"(target={int(ds.labels.sum())}, clutter={int(len(ds.labels)-ds.labels.sum())})")

    cfg_v = dict(cfg_base)
    cfg_v.update({
        "lam_pauc": 0.5, "pauc_alpha": 1e-3, "pauc_buffer_capacity": 2000,
        "save_dir": str(OUT / "pauc_loss"),
        "ckpt_name": "best_model_Full_plus_pAUC.pt",
    })

    t0 = time.time()
    model_v, history_v, test_loader_v, device_v = sim.train(
        cfg_v, ds, model_kwargs={}, verbose_progress=False,
        early_stop_metric="auc_val",
        progress_path=str(OUT / "pauc_progress.json"))
    wall_s = time.time() - t0

    # ── check for training instability (divergence), per the plan's
    #    explicit instruction to report this honestly if it happens ──
    loss_hist = history_v["loss"]
    pauc_hist = history_v["pauc"]
    diverged = (any(not np.isfinite(v) for v in loss_hist) or
                (len(loss_hist) >= 2 and loss_hist[-1] > 3 * loss_hist[0]))

    all_scores, all_labels = [], []
    model_v.eval()
    with torch.no_grad():
        for x, y in test_loader_v:
            score, *_ = model_v(x.to(cfg_v["device"]))
            all_scores.append(score.cpu().numpy())
            all_labels.append(y.numpy())
    scores_v = np.concatenate(all_scores)
    labels_v = np.concatenate(all_labels)

    min_pfa_v = sim.min_representable_pfa(labels_v)
    pfa_report_v = 1e-2 if min_pfa_v <= 1e-2 else round(min_pfa_v, 4)
    pt_auc, lo_auc, hi_auc, _, _ = sim.bootstrap_auc_ci(scores_v, labels_v, n_boot=1000, seed=42)
    pt_pd, lo_pd, hi_pd, _, _ = sim.bootstrap_pd_ci(scores_v, labels_v, pfa=pfa_report_v, n_boot=1000, seed=42)

    stage = {
        "done": True,
        "config_overrides_vs_Full_DU-CCEBM": {
            "lam_pauc": cfg_v["lam_pauc"], "pauc_alpha": cfg_v["pauc_alpha"],
            "pauc_buffer_capacity": cfg_v["pauc_buffer_capacity"],
        },
        "note": ("Same data/splits/training budget as "
                 "run_revision_experiments.py Stage 2 ('matched-scale "
                 "ablation suite'), so 'Full DU-CCEBM' in stage2_ablation "
                 "is the direct comparison point."),
        "n_clutter_generated": N_CLUTTER, "n_target_generated": N_TARGET,
        "epochs_run": len(loss_hist),
        "wall_clock_s": wall_s,
        "training_diverged_or_unstable": bool(diverged),
        "loss_history": loss_hist,
        "pauc_loss_history": pauc_hist,
        "test_n": int(len(labels_v)),
        "reported_pfa_operating_point": pfa_report_v,
        "auc": {"point": pt_auc, "ci95": [lo_auc, hi_auc]},
        "pd":  {"point": pt_pd,  "ci95": [lo_pd, hi_pd]},
    }

    # paired comparison vs the cached "Full DU-CCEBM" checkpoint on the
    # SAME split (same seed/dataset -> deterministic random_split)
    full_ckpt_path = OUT / "best_model_Full_DU-CCEBM.pt"
    if full_ckpt_path.exists():
        ckpt = torch.load(full_ckpt_path, map_location="cpu", weights_only=False)
        model_full = sim.DUCCEBM(L_net=cfg_base["L_net"], N=cfg_base["N"],
                                  L_filt=cfg_base["L_filt"], K_white=cfg_base["K_white"])
        model_full.load_state_dict(ckpt["model"])
        model_full.eval()
        with torch.no_grad():
            all_s, all_l = [], []
            for x, y in test_loader_v:   # identical split object
                s, *_ = model_full(x)
                all_s.append(s.numpy()); all_l.append(y.numpy())
        scores_full = np.concatenate(all_s)
        labels_full = np.concatenate(all_l)
        assert np.array_equal(labels_full, labels_v), "split mismatch"

        pt, lo, hi, sig = sim.bootstrap_auc_difference_ci(
            scores_v, scores_full, labels_full, n_boot=1000, seed=7)
        stage["auc_diff_vs_Full_DU-CCEBM_approx_projection_no_pauc"] = {
            "delta_auc_pauc_variant_minus_full": pt,
            "ci95": [lo, hi],
            "statistically_significant_at_0.05": bool(sig),
        }
        print(f"\n[compare] Full+pAUC vs Full: delta_AUC={pt:+.4f} "
              f"[{lo:+.4f},{hi:+.4f}] significant={sig}")
    else:
        print(f"[warn] {full_ckpt_path} not found, skipping paired comparison.")
        stage["auc_diff_vs_Full_DU-CCEBM_approx_projection_no_pauc"] = None

    master[STAGE_KEY] = stage
    save_master(master)
    print("\n[STAGE 3 (partial-AUC loss) SUMMARY]")
    print(json.dumps({k: v for k, v in stage.items()
                       if k not in ("loss_history", "pauc_loss_history")}, indent=2))

print("\n[DONE]" if master.get(STAGE_KEY, {}).get("done") else "\n[INCOMPLETE]")
