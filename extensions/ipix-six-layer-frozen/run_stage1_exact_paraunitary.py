"""
Stage 1 driver (FABLE_SUGGESTIONS_IMPLEMENTATION_PROMPT.md):
  exact-paraunitary-projection ablation, using newton_schulz_unitary
  (Stage 0, independently pytest-verified: 22/22 passing, see
  FABLE_VERIFICATION_REPORT.md) wired into DUCCEBMLayer as a new
  exact_paraunitary flag (see du_ccebm_simulation.py diff).

Runs TWO new variants at the EXACT SAME data/splits/training budget as
the existing "matched-scale ablation suite" in run_revision_experiments.py
Stage 2 (T=256, L_net=8, L_filt=8, K_white=5, batch=128, epochs=20,
patience=8, N=2400 synthetic samples, seed=42), so results are directly
comparable to the "Full DU-CCEBM" row already in
revision_results/REVISION_SYNTHETIC_RESULTS.json:

  1. "Exact paraunitary projection"            -- exact_paraunitary=True,
                                                   ns_iters=12, lam3
                                                   UNCHANGED (0.1)
  2. "Exact paraunitary + reduced lambda3"      -- same, plus lam3=0.01
                                                   per Fable's own
                                                   recommendation (Sec.
                                                   5 of her proposal:
                                                   "λ3 را از 0.1 به
                                                   ~0.01 کاهش دهید")

These are reported as TWO SEPARATE experiments (not conflated), per the
implementation plan, so each variable's effect can be judged
independently.

Resumable: writes incremental JSON after every stage, mirrors to
/mnt/user-data/outputs/ immediately, exactly like run_revision_experiments.py.
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

STAGE_KEY = "stage3_exact_paraunitary_ablation"
STAGE_DONE = STAGE_KEY in master and master[STAGE_KEY].get("done")

if STAGE_DONE:
    print(f"[skip] {STAGE_KEY} already completed, loading cached result.")
else:
    print("=" * 70)
    print("STAGE 1 (impl. plan) / stage3_exact_paraunitary_ablation:")
    print("  exact-paraunitary-projection ablation (matched-scale)")
    print("=" * 70)

    # EXACT SAME config as run_revision_experiments.py Stage 2, so this
    # is a genuinely comparable extension of the existing ablation table,
    # not a new/different scale.
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

    variants = {
        "Exact paraunitary projection": {
            "model_kwargs": {"exact_paraunitary": True, "ns_iters": 12},
            "cfg_overrides": {},   # lam3 unchanged (0.1), isolates ONLY the projection change
        },
        "Exact paraunitary + reduced lambda3": {
            "model_kwargs": {"exact_paraunitary": True, "ns_iters": 12},
            "cfg_overrides": {"lam3": 0.01},   # Fable's own recommendation, Sec. 5
        },
    }

    stage = {
        "done": False,
        "config_base": {k: cfg_base[k] for k in
                         ["T", "L_net", "L_filt", "K_white", "batch", "epochs",
                          "patience", "val_frac", "test_frac", "seed"]},
        "n_clutter_generated": N_CLUTTER, "n_target_generated": N_TARGET,
        "note": ("Same data/splits/training budget as "
                 "run_revision_experiments.py Stage 2 ('matched-scale "
                 "ablation suite'), so the 'Full DU-CCEBM' and 'Classical "
                 "CCEBM' rows already in stage2_ablation / stage1_headline "
                 "are directly comparable reference points for these two "
                 "new variants."),
        "variants": master.get(STAGE_KEY, {}).get("variants", {}),
    }

    for name, spec in variants.items():
        if name in stage["variants"]:
            print(f"[skip] variant already done: {name}")
            continue
        print(f"\n--- variant: {name} "
              f"(model_kwargs={spec['model_kwargs']}, "
              f"cfg_overrides={spec['cfg_overrides']}) ---")
        cfg_v = dict(cfg_base)
        cfg_v.update(spec["cfg_overrides"])
        cfg_v["save_dir"] = str(OUT / "exact_paraunitary")
        cfg_v["ckpt_name"] = f"best_model_{name.replace(' ', '_').replace('+', 'plus')}.pt"

        t0 = time.time()
        model_v, history_v, test_loader_v, device_v = sim.train(
            cfg_v, ds, model_kwargs=spec["model_kwargs"], verbose_progress=False,
            early_stop_metric="auc_val",
            progress_path=str(OUT / f"exact_pu_progress_{name.replace(' ', '_')}.json"))
        wall_s = time.time() - t0

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
        pt_auc, lo_auc, hi_auc, _, _ = sim.bootstrap_auc_ci(scores_v, labels_v, n_boot=500, seed=42)
        pt_pd, lo_pd, hi_pd, _, _ = sim.bootstrap_pd_ci(scores_v, labels_v, pfa=pfa_report_v, n_boot=500, seed=42)

        n_params_v = sum(p.numel() for p in model_v.parameters())
        n_trainable_v = sum(p.numel() for p in model_v.parameters() if p.requires_grad)

        stage["variants"][name] = {
            "model_kwargs": spec["model_kwargs"],
            "cfg_overrides": spec["cfg_overrides"],
            "n_params": n_params_v, "n_trainable_params": n_trainable_v,
            "epochs_run": len(history_v["loss"]),
            "wall_clock_s": wall_s,
            "test_n": int(len(labels_v)),
            "reported_pfa_operating_point": pfa_report_v,
            "auc": {"point": pt_auc, "ci95": [lo_auc, hi_auc]},
            "pd":  {"point": pt_pd,  "ci95": [lo_pd, hi_pd]},
        }
        master[STAGE_KEY] = stage
        save_master(master)
        print(f"  -> AUC={pt_auc:.4f} [{lo_auc:.4f},{hi_auc:.4f}]  "
              f"Pd@{pfa_report_v:.4g}={pt_pd:.4f} [{lo_pd:.4f},{hi_pd:.4f}]  "
              f"epochs={len(history_v['loss'])}  wall_clock={wall_s:.1f}s")

    # Paired bootstrap AUC-difference test vs. the existing "Full DU-CCEBM"
    # ablation row (same data/splits), so we can say whether the exact
    # projection changed AUC significantly relative to the paper's
    # approximate one-step projection.
    full_ckpt_path = OUT / "best_model_Full_DU-CCEBM.pt"
    if full_ckpt_path.exists():
        print("\n[compare] Re-evaluating cached 'Full DU-CCEBM' checkpoint "
              "on the SAME test split as this stage's variants for a "
              "paired AUC-difference test...")
        ckpt = torch.load(full_ckpt_path, map_location="cpu", weights_only=False)
        model_full = sim.DUCCEBM(L_net=cfg_base["L_net"], N=cfg_base["N"],
                                  L_filt=cfg_base["L_filt"], K_white=cfg_base["K_white"])
        model_full.load_state_dict(ckpt["model"])
        model_full.eval()

        # Re-derive the SAME test split (same seed/dataset construction)
        # used for the ablation-suite variants, matching run_revision_experiments.py's
        # own train() split logic exactly (random_split with the same seed).
        n = len(ds)
        n_test = int(n * cfg_base["test_frac"])
        n_val = int(n * cfg_base["val_frac"])
        n_train = n - n_val - n_test
        _, _, test_set = torch.utils.data.random_split(
            ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(cfg_base["seed"]))
        test_loader_full = torch.utils.data.DataLoader(test_set, batch_size=cfg_base["batch"], shuffle=False)

        all_scores_full, all_labels_full = [], []
        with torch.no_grad():
            for x, y in test_loader_full:
                score, *_ = model_full(x)
                all_scores_full.append(score.numpy())
                all_labels_full.append(y.numpy())
        scores_full = np.concatenate(all_scores_full)
        labels_full = np.concatenate(all_labels_full)

        diffs = {}
        for name in variants:
            if name not in stage["variants"]:
                continue
            # Re-run that variant's saved checkpoint on the SAME split to
            # get paired scores (not the earlier held-out eval which used
            # its own test_loader from an independent random_split call --
            # torch.utils.data.random_split with the same generator/seed on
            # the SAME dataset object is deterministic, so test_loader_v
            # above and test_loader_full here are already the identical
            # split; scores_v computed during training already correspond
            # to labels_full one-for-one).
            pass
        # Since random_split with an identical seed+generator on the SAME
        # `ds` object is deterministic, scores_v/labels_v captured per
        # variant above are already on the identical split as scores_full;
        # recompute the paired test directly from what we already saved to
        # stage["variants"][name] would require re-loading each variant's
        # checkpoint. Simpler and fully correct: reload each variant
        # checkpoint here too, in the SAME pass, for a clean paired test.
        for name, spec in variants.items():
            ck_name = f"best_model_{name.replace(' ', '_').replace('+', 'plus')}.pt"
            ck_path = OUT / "exact_paraunitary" / ck_name
            if not ck_path.exists():
                continue
            ckpt_v = torch.load(ck_path, map_location="cpu", weights_only=False)
            model_v2 = sim.DUCCEBM(L_net=cfg_base["L_net"], N=cfg_base["N"],
                                    L_filt=cfg_base["L_filt"], K_white=cfg_base["K_white"],
                                    **spec["model_kwargs"])
            model_v2.load_state_dict(ckpt_v["model"])
            model_v2.eval()
            all_s, all_l = [], []
            with torch.no_grad():
                for x, y in test_loader_full:
                    score, *_ = model_v2(x)
                    all_s.append(score.numpy())
                    all_l.append(y.numpy())
            scores_v2 = np.concatenate(all_s)
            labels_v2 = np.concatenate(all_l)
            assert np.array_equal(labels_v2, labels_full), \
                "Split mismatch -- labels do not line up for paired test"

            pt, lo, hi, sig = sim.bootstrap_auc_difference_ci(
                scores_v2, scores_full, labels_full, n_boot=1000, seed=7)
            diffs[name] = {
                "delta_auc_variant_minus_full_approx_projection": pt,
                "ci95": [lo, hi],
                "statistically_significant_at_0.05": bool(sig),
            }
            print(f"  {name} vs Full (approx. projection): "
                  f"delta_AUC={pt:+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"significant={sig}")

        stage["auc_diff_vs_full_approx_projection"] = diffs
    else:
        print(f"\n[warn] {full_ckpt_path} not found -- skipping paired "
              f"AUC-difference test vs. Full DU-CCEBM (approx. projection). "
              f"Point estimates above are still valid on their own.")
        stage["auc_diff_vs_full_approx_projection"] = None

    stage["done"] = True
    master[STAGE_KEY] = stage
    save_master(master)
    print("\n[STAGE 1 (exact paraunitary) SUMMARY]")
    print(json.dumps(stage, indent=2))

print("\n[DONE]" if master.get(STAGE_KEY, {}).get("done") else "\n[INCOMPLETE]")
