# DU-CCEBM — Maritime Radar Reproducibility

**Sole author:** Hamzeh Ghahramani, Department of Electrical Engineering, Islamic Azad University,
Hormozgan, Iran.  Research software release candidate `v1.0.0`; **not yet tagged or archived**.

This is a **new, clean Git repository with no private Git history**. It is constructed from verified
scientific assets previously staged privately. Its results represent **three different model
configurations**; they are not a single interchangeable detector or study.

| Study | Scope | Local verification |
|---|---|---|
| Frozen two-layer IPIX long-CPI | Frozen weights, code, score-level evidence, original baseline timing | Original source SHA-256 manifest, `scripts/recompute_longcpi.py`, smoke test |
| Per-window four-layer NetRAD/IPIX | 21 NetRAD per-acquisition score snapshots, 8 IPIX acquisition summaries | 8 numeric CSVs regenerated from packaged snapshots |
| Frozen six-layer IPIX | Training/inference source and original frozen checkpoint | Archived 240 + 22,416 scores reproduced from 14 raw IPIX CDFs; no independent full retraining |

## Minimal verification

```bash
python -m pip install -r requirements.txt
python scripts/verify_public_package.py
python scripts/verify_release.py
python scripts/recompute_longcpi.py --out outputs/longcpi
python scripts/smoke_inference.py
(cd extensions/netrad-ipix-4layer && python verify_release_candidate.py)
(cd extensions/ipix-six-layer-frozen && python verify_sixlayer.py)
```

PyTorch/NumPy/Matplotlib and scikit-learn are needed for six-layer verification; see the
extension's README for tests and optional raw IPIX re-scoring. For standalone four-layer tests
install `pytest`. Raw NetRAD data and IPIX CDF are **not** redistributed.

## What the study establishes—and does not

Two-layer inference-time findings are **specific to the frozen two-layer benchmark**. The newer
four-layer version fits/optimizes each window: earlier speed results do not apply to that method.
NetRAD empirical Pfa=0.01 uses H0 threshold calibration, not independent held-out CFAR validation.
Different detector baselines sometimes use different available channels; read the individual
protocol documents before comparing detection probabilities. The IPIX four-layer “ROC-style”
figures summarize operating points and do not represent empirical raw-score ROC functions.
Six-layer AUCs (0.526175 mixed, 0.804293 genuine-only) come from the **historical six-layer**
experiment, not the frozen two-layer or four-layer data. The genuine-only set is not an entirely
held-out acquisition. Retraining-from-scratch has not been independently verified.

## Licensing, citations and availability

- Main two-layer and six-layer author-owned **software**: BSD-3-Clause (`LICENSE`).
- Four-layer NetRAD and IPIX author-owned software: BSD-2-Clause (`extensions/netrad-ipix-4layer/*/LICENSE`).
- Numeric results, model weights and figures: see `LICENSE_SCOPE.md`; software licenses do not
  automatically license third-party raw datasets or original publisher material.
- Original raw IPIX/NetRAD recordings and older potentially publisher-derived manuscript figures
  are excluded deliberately. Bibliographic references to other published works retain proper credits.
- AI assistance in software/research drafting is disclosed in `docs/AI_ASSISTED_DEVELOPMENT.md`.

**No Zenodo DOI has been assigned as of this candidate build.** Do not cite an invented DOI.
Repository intended for this clean snapshot: `https://github.com/Hamzeh62/DU-CCEBM-Reproducibility`.
