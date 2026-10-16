# Frozen six-layer IPIX DU-CCEBM reference

**Sole author: Hamzeh Ghahramani.** The sole-author source-header correction changes comments only.
This is a separate frozen six-layer model, not the same trained checkpoint or experiment as the
frozen two-layer or per-window optimized four-layer code.

- Model checkpoint `checkpoints/real_A_best_model.pt` has SHA-256
  `5d450c073c22c3cb9aeca8b49fe837549f3921d347439b3db268f3234981fb01`.
- Exact original configuration: `L_net=6`, `L_filt=6`, `K_white=4`, `T=128`, seed 42.
- Archived results: 240 mixed-set examples, 22,416 genuine-only examples, reference DU AUCs
  0.5261752512 and 0.8042931034 respectively.
- The genuine-only set is *not* an independent held-out acquisition/file.
- The original 14 IPIX NetCDF recordings are not redistributed.

```bash
python verify_sixlayer.py
python -m pytest -q
python verify_sixlayer.py --raw-ipix-dir /path/to/14/original/ipix/cdf
```

`verify_sixlayer.py` checks the shipped-file SHA-256 manifest, the 100-entry strict checkpoint
load, and archived score/AUC arrays; optional raw CDF mode recomputes and compares scores. An
independent audit using the supplied 14 files confirmed all 240 + 22,416 scores within 1e-6.
This is a **frozen-checkpoint inference reproduction**, not retraining from scratch.

The old manuscript draft and its figures were intentionally omitted to avoid conflating a prior
submission with the present code deposit and to reduce rights ambiguity. The original source
ZIP remains in the author's private archive; provenance of the provided scientific files is
recorded in `SOURCE_PROVENANCE.json`.
