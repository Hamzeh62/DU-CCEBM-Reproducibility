# NetRAD — DU-CCEBM four-layer detection extension (nine-arm evidence)

21 Path-A acquisitions (node N3), 128-sample windows, Doppler offset +21 Hz,
400 H0 and 80 H1 per acquisition/SCR, SCR levels -25, -20, -15, -10, -5 dB.
Data are target-injected; per-acquisition operating point Pfa=0.01 is computed
on the 400 H0 **calibration samples**. At the registered threshold, 4/400
H0 score statistics exceed threshold (empirical calibration Pfa=0.01).

The proposed arm is a **four-layer model optimizing each window** and is not
interchangeable with earlier frozen two-layer or six-layer results.

Run `python -m pytest tests -q` followed by `python repro_norival_du.py`.
This regenerates five CSV tables in `data_norival_DU/` and the three
selected figures in `figs_norival_DU/` from 21 cleaned statistic-score JSON
snapshots in `data/checkpoints_du/`.

Eight comparators: CCEBM-PU/WL/DP, STFICA, Kelly GLRT, AMF, ANMF, CA-CFAR.
CERBM/DU-CERBM numeric checkpoints were deleted from this publication copy;
`code/cerbm.py` is a dependency imported by the classical CCEBM module, not
an evaluated comparator. Original full checkpoints stay in the author's archive.

The optional source `du_netroc_add_duccebm.py` documents the acquisition-level
raw workflow but depends on NetRAD reader and data resources **not shipped**
with this compact release. It is not an offline, end-to-end raw-data rerunner.
The numeric AUC/Pd summaries are in `data_norival_DU/` and must be used for
scientific claims, rather than manually transcribed secondary descriptions.

Source license in `LICENSE` (BSD-2-Clause), with dataset distribution terms
applying separately. See ../docs/PROTOCOLS_AND_FAIRNESS.md.
