# Registered experiment protocols, inference scope and scientific claims

| Dimension | NetRAD | IPIX |
|---|---|---|
| Study | Path-A, node N3 | Sea-clutter recordings, eight acquisitions |
| Independent acquisition units | 21 | 8 |
| Decision length | M = 128 | M = 10/32/128 saved; M = 128 figures |
| Doppler offset | +21 Hz to local clutter centroid | +21 / +60 Hz to clutter centroid |
| SCR (dB) | -25, -20, -15, -10, -5 | -30 to +5 in steps of 5 |
| H0/H1 sample counts | 400 H0, 80 H1 per SCR and acquisition | 1500 H0, 150 H1 per SCR and acquisition |
| Nominal Pfa | 0.01; 4/400 empirically on threshold-calibration H0 | 0.01, per-acquisition calibrated |
| Source result snapshot | Per-window score arrays by method and condition, 21 JSONs | Per-file Pd summaries across the design grid, one merged JSON |
| AUC evidence | Mann-Whitney of saved H0/H1 score pairs | No raw-score AUC in the included summary; use detection-power tables |
| Summary statistic | Mean AUC/Pd over 21 acquisitions; paired acquisition-level bootstrap | Median Pd over eight files; paired sign test over file-level summaries |
| Model | Four unfolded layers, warm start, per-window optimization | Four unfolded layers, warm start, per-window optimization |
| Conventional inputs | Registered CUT/reference-cell configuration | HH for certain conventional arms, HH/VV for some BSS arms |
| False-alarm interpretation | Calibration-set empirical value; no extrapolation to independent CFAR without data | Nominal per-file threshold calibration |

## Comparison design

The experiment comparisons follow the registered method-specific input conventions. The resulting head-to-head tables apply to those **registered configurations**; they should not be described as an all-arms equal-channel/equal-information experiment unless a separately matched-input analysis is supplied. Attribution of a later improvement should match the tested model and input representation. The earlier two-layer runtime advantage is a different benchmark and is not re-assigned to the present four-layer per-window optimizer.

## Statistical multiplicity

NetRAD effect tables report 40 DU-to-eight-rival effects across five SCRs, plus the recorded Holm correction. For IPIX there are 16 paired comparisons (2 offsets × 8 rivals); under Holm correction across all 16, no IPIX difference remains below 0.05. If a smaller family was pre-specified, provide the pre-registration before making claims based on a different correction family.

## Numeric versus visual reproducibility

The eight combined CSV tables are reproducible byte-for-byte from the packaged release JSONs (verified after sanitization). PDFs are vector images subject to library-generated metadata and need not have the same binary hash to represent the same curves. Displayed IPIX `ROC-style` envelopes are descriptive plots of the saved operating-point summaries; they do not represent score-based empirical ROC functions.
