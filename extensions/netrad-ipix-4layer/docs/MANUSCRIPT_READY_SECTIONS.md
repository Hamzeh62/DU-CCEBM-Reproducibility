# Manuscript-ready scope statement and reproducibility text (draft)

The text below describes the actual, registered four-layer study. Before incorporation into the Signal Processing submission, place it in a distinctly labeled experimental subsection so that its results are not confused with the historical six-layer and frozen two-layer experiments.

## Benchmark setup

The four-layer DU-CCEBM variant was evaluated on two separate sea-clutter radar benchmarks with acquisition-level statistical aggregation. The NetRAD Path-A/N3 study used 21 acquisitions, 128-pulse decision windows, a Doppler offset of +21 Hz from the estimated clutter centroid, and a SCR range from -25 to -5 dB. For each acquisition, the saved scores comprise 400 H0 windows and 80 H1 windows at each SCR. Per-acquisition thresholds were calibrated at a nominal false-alarm probability of 0.01 using the H0 calibration scores (4/400 at the registered threshold). AUC was evaluated from saved target-absent and target-present scores and paired acquisition-level resampling was used for statistical comparison.

The independent IPIX experiment covered eight radar acquisitions and decision windows M=10, 32 and 128, with the reported power figures using M=128. Two Doppler offsets (+21/+60 Hz) and eight SCR levels from -30 to +5 dB were evaluated. Each acquisition and experimental condition included 1,500 H0 and 150 H1 windows in the recorded protocol. Detection probabilities were summarized across acquisitions using their median values. Some conventional reference detectors used HH-only observations, while polarimetric BSS approaches used HH/VV channels; thus the results characterize the methods under the *registered detector-specific observation models* rather than proving equivalence of input information.

## NetRAD results

| SCR (dB) | DU-CCEBM AUC | CCEBM-Full-PU AUC | Kelly GLRT AUC | DU-CCEBM Pd @ nominal Pfa 0.01 | CCEBM-Full-PU Pd |
|---:|---:|---:|---:|---:|---:|
| -25 | 0.6815 | 0.6834 | 0.5851 | 0.0399 | 0.0429 |
| -20 | 0.7955 | 0.7901 | 0.6685 | 0.1238 | 0.1196 |
| -15 | 0.9028 | 0.8990 | 0.7805 | 0.3071 | 0.2905 |
| -10 | 0.9675 | 0.9634 | 0.8891 | 0.5946 | 0.5280 |
| -5 | 0.9902 | 0.9874 | 0.9582 | 0.8185 | 0.7536 |

Across the five SCR levels, DU-CCEBM's mean AUC was above classical CCEBM-Full-PU at four levels and slightly below it at -25 dB. The largest improvement over the classical CCEBM reference was in the detection probability at -10 and -5 dB for the registered calibration operating point. The results support competitive detection performance for the four-layer variant under the specified NetRAD protocol, but do not imply a consistent AUC advantage at every operating condition.

## IPIX results and multiplicity

The IPIX power summaries reproduce from the bundled eight-acquisition JSON. For the acquisition-paired median-over-SCR test, the DU-CCEBM to AMF comparison has seven positive acquisitions of eight at +21 Hz (unadjusted two-sided sign p=0.0703) and eight of eight at +60 Hz (p=0.0078). At +60 Hz the DU-CCEBM to CCEBM-PU comparison is zero positive of eight (mean paired difference DU minus PU=-0.0433; p=0.0078). For a family including all 16 IPIX offset-by-rival tests, no p value survives Holm correction at the 0.05 family-wise level. No unconditional improvement in detection probability over all classical CCEBM variants is claimed. The IPIX 'ROC-style' panels are descriptive envelopes constructed from stored calibrated operating-point summaries, not individual-window empirical ROC curves.

## Computation and speed claim scope

Separate prior studies measured the inference-time advantage for the frozen **two-layer** DU-CCEBM implementation. Those measurements remain valid for the original tested configuration but are **not** ascribed to the present **four-layer per-window-optimized** implementation without a matched timing benchmark. The current NetRAD/IPIX extension is used to assess detector performance rather than to re-establish the historical two-layer speedup.

## Code and data availability

The archived numeric summary tables and figures are reproducible offline from the score-level NetRAD checkpoint snapshots and per-acquisition IPIX sweep results included in the code supplement. The eight generated CSVs were verified to match the source archive byte-for-byte, and all retained per-method score and threshold entries were checked to be unchanged after removal of excluded CERBM-family result fields. The full original radar measurements are not redistributed. Raw-signal re-execution was performed by the researcher using the original datasets and environment; the compact release documents that provenance but does not independently provide all third-party inputs or parent-project modules needed for a raw-to-results rerun.

## AI-assisted implementation disclosure

The research software was developed with the assistance of Anthropic Claude (classical CCEBM modules) and OpenAI ChatGPT (DU extensions, testing, documentation and reproducibility preparation). The author supervised the algorithms, performed the research and numerical verification, and takes responsibility for the resulting source code, analysis and scientific claims. A separate manuscript-preparation declaration should reflect any generative AI assistance used to draft or edit the article text.
