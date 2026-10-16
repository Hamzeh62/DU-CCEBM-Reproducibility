"""
Real-IPIX-data driver, Part B: ANMF baseline + DU-CCEBM evaluation on
REAL McMaster IPIX data, with bootstrap 95% CIs and a paired
AUC-difference significance test -- this completes the "Table VI" (real
data) portion of the implementation plan that Stage 2 explicitly could
not run earlier this session (no .cdf files were available then).

Two evaluations are produced, exactly following the honesty convention
already established in MASTER_REAL_RESULTS_original_session.json's own
"genuine_vs_synthetic_target_breakdown":

  (1) "mixed" -- the standard held-out test split from real_A_train.py's
      training run (240 samples): a mix of genuinely-recorded IPIX
      target/clutter windows AND synthetically-injected-target-on-real-
      clutter windows (see IPIXDataset._load_ipix /
      _inject_targets_dualpol). This is the SAME data composition
      convention used throughout this codebase's existing real-data
      results, so it is directly comparable to them.

  (2) "genuine_only" -- ONLY windows that are genuinely recorded IPIX
      target returns (label=1, from the documented target_cell) or
      genuinely recorded clutter (label=0, from reference cells), with
      NO synthetic injection at all. This is the closest thing to a
      "real Table VI" this codebase can produce and is the number that
      should be trusted most when judging real-world performance, at
      the cost of a much smaller n (and correspondingly wider CIs) --
      exactly the same caveat MASTER_REAL_RESULTS_original_session.json
      already flagged for its own n=45 genuine-target breakdown.
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

if "real_A_train_du_ccebm" not in master or not master["real_A_train_du_ccebm"].get("done"):
    print("[error] real_A_train_du_ccebm not done yet -- run run_real_A_train.py first.")
    sys.exit(1)

STAGE_KEY = "real_B_anmf_evaluation"
if STAGE_KEY in master and master[STAGE_KEY].get("done"):
    print(f"[skip] {STAGE_KEY} already completed.")
else:
    a = master["real_A_train_du_ccebm"]
    cfg = dict(sim.CFG)
    cfg.update(a["config"])
    cfg.update({
        "ipix_dir": "ipix_data",
        "save_dir": str(OUT / "real_du_ccebm"),
        "device": "cpu",
    })

    print("=" * 70)
    print("REAL-B: ANMF baseline + DU-CCEBM evaluation on real IPIX data")
    print("=" * 70)

    # ── Rebuild the EXACT same dataset/subset/split as real_A_train.py ──
    sim.set_seed(cfg["seed"])
    dataset_full = sim.IPIXDataset(cfg, mode="train")
    np.random.seed(42)
    n_use = a["n_samples_used_subset"]
    idx = np.random.choice(len(dataset_full), size=n_use, replace=False)
    dataset = torch.utils.data.Subset(dataset_full, idx)

    n = len(dataset)
    n_test = int(n * cfg["test_frac"])
    n_val = int(n * cfg["val_frac"])
    n_train = n - n_val - n_test
    _, _, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg["seed"]))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg["batch"], shuffle=False)
    print(f"[data] mixed test split: n={len(test_set)}")

    # ── Load the just-trained real-data DU-CCEBM checkpoint ─────────────
    ckpt = torch.load(a["checkpoint_path"], map_location="cpu", weights_only=False)
    model = sim.DUCCEBM(L_net=cfg["L_net"], N=cfg["N"], L_filt=cfg["L_filt"], K_white=cfg["K_white"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    def extract_hh_vv_scores(loader):
        all_scores_du, all_labels = [], []
        all_hh_re, all_hh_im, all_vv_re, all_vv_im = [], [], [], []
        with torch.no_grad():
            for x, y in loader:
                score, *_ = model(x)
                all_scores_du.append(score.numpy())
                all_labels.append(y.numpy())
                all_hh_re.append(x[:, 0, 0].numpy()); all_hh_im.append(x[:, 1, 0].numpy())
                all_vv_re.append(x[:, 0, 1].numpy()); all_vv_im.append(x[:, 1, 1].numpy())
        scores_du = np.concatenate(all_scores_du)
        labels = np.concatenate(all_labels)
        hh = np.concatenate(all_hh_re) + 1j * np.concatenate(all_hh_im)
        vv = np.concatenate(all_vv_re) + 1j * np.concatenate(all_vv_im)
        return scores_du, labels, hh, vv

    def run_anmf_and_compare(scores_du, labels, hh, vv, tag):
        n_t = len(labels)
        print(f"\n--- [{tag}] n={n_t} (target={int(labels.sum())}, "
              f"clutter={int(n_t-labels.sum())}) ---")
        clutter_idx = np.where(labels == 0)[0]
        if len(clutter_idx) < 10:
            print(f"  [warn] too few clutter samples ({len(clutter_idx)}) to "
                  f"estimate R_hat reliably; skipping ANMF for this subset.")
            return None
        clutter_pool = [np.stack([hh[i], vv[i]]) for i in clutter_idx]
        n_calib = min(200, len(clutter_pool))
        R_hat, R_inv = sim.estimate_polarimetric_covariance(
            clutter_pool, n_calib=n_calib, diag_loading=1e-2, seed=42)
        rho = R_hat[0, 1].real / np.sqrt(R_hat[0, 0].real * R_hat[1, 1].real)
        print(f"  [ANMF] R_hat from {n_calib} genuine-clutter windows, HH/VV corr={rho:.3f}")

        X = np.stack([hh, vv], axis=1)
        scores_anmf = sim.anmf_detect_batch(X, R_inv)

        min_pfa = sim.min_representable_pfa(labels)
        pfa_report = 1e-2 if min_pfa <= 1e-2 else round(min_pfa, 4)

        def auc_ci(s):
            pt, lo, hi, _, nf = sim.bootstrap_auc_ci(s, labels, n_boot=1000, seed=42)
            return {"point": pt, "ci95": [lo, hi], "n_boot_failed": int(nf)}

        def pd_ci(s):
            pt, lo, hi, _, nf = sim.bootstrap_pd_ci(s, labels, pfa=pfa_report, n_boot=1000, seed=42)
            return {"point": pt, "ci95": [lo, hi], "n_boot_failed": int(nf)}

        # CA-CFAR too (cheap, same convention as full_evaluation)
        amp = np.abs(hh)
        scores_cfar = amp.mean(-1)
        scores_cfar = (scores_cfar - scores_cfar.min()) / (scores_cfar.max() - scores_cfar.min() + 1e-10)

        result = {
            "n": int(n_t), "n_target": int(labels.sum()), "n_clutter": int(n_t - labels.sum()),
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
        print(f"  DU-CCEBM AUC={result['DU-CCEBM']['auc']['point']:.4f}  "
              f"ANMF AUC={result['ANMF']['auc']['point']:.4f}  "
              f"CA-CFAR AUC={result['CA-CFAR']['auc']['point']:.4f}")
        print(f"  DU-CCEBM vs ANMF: delta_AUC={pt:+.4f} [{lo:+.4f},{hi:+.4f}] significant={sig}")
        return result

    # ── (1) mixed test split ─────────────────────────────────────────────
    scores_du_mix, labels_mix, hh_mix, vv_mix = extract_hh_vv_scores(test_loader)
    res_mixed = run_anmf_and_compare(scores_du_mix, labels_mix, hh_mix, vv_mix, "mixed test split")

    # ── (2) genuine-only pool (no synthetic injection) ──────────────────
    print("\n[data] Extracting GENUINE-ONLY pool (bypassing synthetic injection)...")
    orig_inject = dataset_full._inject_targets_dualpol
    dataset_full._inject_targets_dualpol = lambda clutter_pairs: ([], [])
    files = sorted(Path(cfg["ipix_dir"]).glob("*.cdf"))
    genuine_samples, genuine_labels = dataset_full._load_ipix(files)
    dataset_full._inject_targets_dualpol = orig_inject   # restore
    genuine_labels = np.asarray(genuine_labels, dtype=np.float32)
    print(f"[data] genuine-only pool: n={len(genuine_labels)} "
          f"(target={int(genuine_labels.sum())}, clutter={int(len(genuine_labels)-genuine_labels.sum())})")

    genuine_hh = np.stack([s[0] for s in genuine_samples])
    genuine_vv = np.stack([s[1] for s in genuine_samples])
    genuine_x = np.zeros((len(genuine_samples), 2, 2, cfg["T"]), dtype=np.float32)
    genuine_x[:, 0, 0] = genuine_hh.real; genuine_x[:, 1, 0] = genuine_hh.imag
    genuine_x[:, 0, 1] = genuine_vv.real; genuine_x[:, 1, 1] = genuine_vv.imag
    with torch.no_grad():
        scores_du_genuine = model(torch.from_numpy(genuine_x))[0].numpy()

    res_genuine = run_anmf_and_compare(scores_du_genuine, genuine_labels,
                                        genuine_hh, genuine_vv, "GENUINE-ONLY (no synthetic injection)")

    stage = {
        "done": True,
        "note": ("'mixed' = held-out test split from real_A_train.py (mix of "
                 "genuine + synthetically-injected-on-real-clutter samples, "
                 "same convention as all other real-data results in this "
                 "codebase). 'genuine_only' = ALL genuinely-recorded IPIX "
                 "target/clutter windows from the 14 files, zero synthetic "
                 "injection -- the most trustworthy real-data number, at the "
                 "cost of a much smaller and possibly still class-imbalanced n."),
        "mixed_test_split": res_mixed,
        "genuine_only_full_pool": res_genuine,
    }
    master[STAGE_KEY] = stage
    save_master(master)
    print("\n[REAL-B SUMMARY]")
    print(json.dumps(stage, indent=2))
