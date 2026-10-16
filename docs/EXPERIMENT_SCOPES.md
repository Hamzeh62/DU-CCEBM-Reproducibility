# Experiment scope and interpretation matrix

| Property | Frozen two-layer | Per-window four-layer | Frozen six-layer |
|---|---|---|---|
| Data | IPIX | NetRAD and IPIX | IPIX |
| Layers | 2 | 4 | 6 |
| Checkpoint | Prior-validation-selected frozen checkpoint | Per-window fitting (not same frozen model) | `real_A_best_model.pt` |
| Reproduced | Score aggregation and local smoke tests | Eight numeric CSV files from 21 NetRAD snapshots and 8 IPIX summaries | 22,656 archived scores re-scored from 14 raw IPIX acquisitions |
| Training | Not part of a new training run | Per-window optimizer in code | Full training from scratch not independently reproduced |
| Timing claims | Specific to frozen two-layer implementation | No inheritance of speed advantage | No automatic inheritance of two-layer timing |
| Pfa | Prespecified nominal calibration with temporal-holdout diagnostics | Empirical NetRAD calibration-H0 Pfa=0.01 | Historical evaluation; see archived reports |

The six-layer genuine-only evaluation does not mean an entirely independent held-out acquisition.
Four-layer IPIX 'ROC-style' plots are operating-point summaries. See the full protocol files for
input-polarization and baseline differences and multiple-comparison treatment.
