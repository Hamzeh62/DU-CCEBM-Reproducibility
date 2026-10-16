"""
Stage 2 driver (FABLE_SUGGESTIONS_IMPLEMENTATION_PROMPT.md):
  Add ANMF (Adaptive Normalized Matched Filter) as a baseline, evaluated
  on the SAME test splits already used for the headline (n=600) and
  matched-scale ablation-suite (n=240) synthetic runs, with bootstrap
  95% CIs and a paired-bootstrap AUC-difference significance test
  against DU-CCEBM -- this last part directly answers the user's core
  question ("Pd بهتر از ANMF شده؟").

REAL DATA: confirmed absent this session (no *.cdf files anywhere under
/mnt or elsewhere -- see verification immediately before this script was
written). The real-data portion of this stage (Table VI in the
implementation plan) is therefore explicitly NOT run; only the synthetic
portion (Table VII-equivalent) is produced, exactly as
FABLE_SUGGESTIONS_IMPLEMENTATION_PROMPT.md Stage 2 instructs for this
contingency ("اگر داده‌ی واقعی موجود بود").

Resumable: incremental JSON, mirrored to /mnt/user-data/outputs/.
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

STAGE_KEY = "stage4_anmf_baseline"
if STAGE_KEY in master and master[STAGE_KEY].get("done"):
    print(f"[skip] {STAGE_KEY} already completed.")
else:
    stage = {"done": False, "real_data_available_this_session": False,
             "note": ("Real IPIX .cdf files were confirmed absent from "
                       "this session's uploads (verified by filesystem "
                       "search immediately before this stage ran); only "
                       "the synthetic-data portion of the ANMF baseline "
                       "is produced here, per the implementation plan's "
                       "explicit instruction for this contingency."),
             "runs": {}}

    def evaluate_one_run(run_name, ckpt_path, model_class_kwargs,
                          cfg_data, n_clutter, n_target, pfa_report_override=None):
        """Rebuild the exact dataset used for a prior run (same seed,
        same n_clutter/n_target -> IPIXDataset._synthetic_dataset is a
        deterministic function of its RNG state, which set_seed(seed)
        resets identically every time), re-derive the exact same
        train/val/test split train() would produce (same seed +
        generator), reload the cached DU-CCEBM checkpoint, compute ANMF
        on the identical test split, and run the paired AUC-difference
        test.
        """
        print(f"\n--- {run_name} ---")
        sim.set_seed(cfg_data["seed"])
        ds = sim.IPIXDataset.__new__(sim.IPIXDataset)
        ds.cfg, ds.mode, ds.T = cfg_data, "train", cfg_data["T"]
        ds.pols, ds.P_ref = cfg_data["pols"], cfg_data["P_ref"]
        ds.samples, ds.labels = ds._synthetic_dataset(n_clutter=n_clutter, n_target=n_target)
        print(f"[data] {len(ds)} samples "
              f"(target={int(ds.labels.sum())}, clutter={int(len(ds.labels)-ds.labels.sum())})")

        n = len(ds)
        n_test = int(n * cfg_data["test_frac"])
        n_val = int(n * cfg_data["val_frac"])
        n_train = n - n_val - n_test
        _, _, test_set = torch.utils.data.random_split(
            ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(cfg_data["seed"]))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg_data["batch"], shuffle=False)

        if not Path(ckpt_path).exists():
            print(f"[warn] checkpoint {ckpt_path} not found, skipping this run.")
            return None

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = sim.DUCCEBM(L_net=cfg_data["L_net"], N=cfg_data["N"],
                             L_filt=cfg_data["L_filt"], K_white=cfg_data["K_white"],
                             **model_class_kwargs)
        model.load_state_dict(ckpt["model"])
        model.eval()

        all_scores_du, all_labels = [], []
        all_hh_re, all_hh_im, all_vv_re, all_vv_im = [], [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                score, *_ = model(x)
                all_scores_du.append(score.numpy())
                all_labels.append(y.numpy())
                all_hh_re.append(x[:, 0, 0].numpy()); all_hh_im.append(x[:, 1, 0].numpy())
                all_vv_re.append(x[:, 0, 1].numpy()); all_vv_im.append(x[:, 1, 1].numpy())
        scores_du = np.concatenate(all_scores_du)
        labels = np.concatenate(all_labels)
        hh_all = np.concatenate(all_hh_re) + 1j * np.concatenate(all_hh_im)
        vv_all = np.concatenate(all_vv_re) + 1j * np.concatenate(all_vv_im)
        n_test_actual = len(labels)
        print(f"[test set] n={n_test_actual} "
              f"(target={int(labels.sum())}, clutter={int(n_test_actual - labels.sum())})")

        # ── ANMF ─────────────────────────────────────────────────────
        # R_hat estimated from a pool of PURE-CLUTTER windows drawn from
        # the TEST set itself (secondary data role, independent of any
        # given CUT being scored -- same convention as
        # calibrate_ccebm_reference / clutter_pool used for CCEBM above,
        # which this codebase already treats as legitimate "reference
        # data" rather than training-set leakage, since it never uses
        # target-bearing samples or labels beyond "is this pure
        # clutter").
        clutter_idx = np.where(labels == 0)[0]
        clutter_pool_anmf = [np.stack([hh_all[i], vv_all[i]]) for i in clutter_idx]
        n_calib = min(200, len(clutter_pool_anmf))
        R_hat, R_inv = sim.estimate_polarimetric_covariance(
            clutter_pool_anmf, n_calib=n_calib, diag_loading=1e-2, seed=42)
        print(f"[ANMF] R_hat estimated from {n_calib} pure-clutter windows "
              f"(HH/VV corr={R_hat[0,1].real/np.sqrt(R_hat[0,0].real*R_hat[1,1].real):.3f})")

        X_all = np.stack([hh_all, vv_all], axis=1)          # (n_test, 2, T)
        scores_anmf = sim.anmf_detect_batch(X_all, R_inv)

        # ── operating point / bootstrap CIs (same convention as Stage 1)
        min_pfa = sim.min_representable_pfa(labels)
        pfa_report = pfa_report_override or (1e-2 if min_pfa <= 1e-2 else round(min_pfa, 4))

        def auc_ci(scores):
            pt, lo, hi, _, nfail = sim.bootstrap_auc_ci(scores, labels, n_boot=1000, seed=42)
            return {"point": pt, "ci95": [lo, hi], "n_boot_failed": int(nfail)}

        def pd_ci(scores):
            pt, lo, hi, _, nfail = sim.bootstrap_pd_ci(scores, labels, pfa=pfa_report, n_boot=1000, seed=42)
            return {"point": pt, "ci95": [lo, hi], "n_boot_failed": int(nfail)}

        result = {
            "test_n": int(n_test_actual),
            "n_calib_clutter_for_R_hat": int(n_calib),
            "reported_pfa_operating_point": pfa_report,
            "min_representable_nonzero_pfa": min_pfa,
            "DU-CCEBM":  {"auc": auc_ci(scores_du),   "pd": pd_ci(scores_du)},
            "ANMF":      {"auc": auc_ci(scores_anmf), "pd": pd_ci(scores_anmf)},
        }

        pt, lo, hi, sig = sim.bootstrap_auc_difference_ci(
            scores_du, scores_anmf, labels, n_boot=1000, seed=7)
        result["auc_diff_DU_CCEBM_minus_ANMF"] = {
            "point": pt, "ci95": [lo, hi],
            "statistically_significant_at_0.05": bool(sig),
        }
        print(f"  DU-CCEBM AUC={result['DU-CCEBM']['auc']['point']:.4f}  "
              f"ANMF AUC={result['ANMF']['auc']['point']:.4f}  "
              f"delta={pt:+.4f} [{lo:+.4f},{hi:+.4f}] significant={sig}")
        print(f"  DU-CCEBM Pd@{pfa_report:.4g}={result['DU-CCEBM']['pd']['point']:.4f}  "
              f"ANMF Pd@{pfa_report:.4g}={result['ANMF']['pd']['point']:.4f}")
        return result

    # ── Run 1: headline scale (n=3000+3000 generated, 35-epoch budget) ──
    cfg_headline = dict(sim.CFG)
    cfg_headline.update({
        "T": 256, "L_net": 8, "L_filt": 8, "K_white": 5,
        "batch": 128, "epochs": 35, "patience": 12,
        "val_frac": 0.15, "test_frac": 0.10, "seed": 42,
    })
    res_headline = evaluate_one_run(
        "headline (n_gen=3000+3000, matches stage1_headline)",
        OUT / "best_model_headline.pt", {},
        cfg_headline, n_clutter=3000, n_target=3000)
    if res_headline is not None:
        stage["runs"]["headline_synthetic"] = res_headline
        save_master({**master, STAGE_KEY: stage})

    # ── Run 2: matched-scale ablation suite (n=1200+1200, matches stage2) ─
    cfg_ablation = dict(sim.CFG)
    cfg_ablation.update({
        "T": 256, "L_net": 8, "L_filt": 8, "K_white": 5,
        "batch": 128, "epochs": 20, "patience": 8,
        "val_frac": 0.15, "test_frac": 0.10, "seed": 42,
    })
    res_ablation = evaluate_one_run(
        "matched-scale ablation suite (n_gen=1200+1200, matches stage2_ablation)",
        OUT / "best_model_Full_DU-CCEBM.pt", {},
        cfg_ablation, n_clutter=1200, n_target=1200)
    if res_ablation is not None:
        stage["runs"]["ablation_scale_synthetic"] = res_ablation
        save_master({**master, STAGE_KEY: stage})

    stage["done"] = True
    master[STAGE_KEY] = stage
    save_master(master)
    print("\n[STAGE 2 (ANMF) SUMMARY]")
    print(json.dumps(stage, indent=2))

print("\n[DONE]" if master.get(STAGE_KEY, {}).get("done") else "\n[INCOMPLETE]")
