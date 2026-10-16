# IPIX — four-layer DU-CCEBM detection extension (nine arms)

8 IPIX acquisitions. The validated sweep contains M=10,32,128; publication
tables/figures use M=128, two offsets +21/+60 Hz, SCR=-30 to +5 dB by 5 dB,
1500 H0 and 150 H1 windows per acquisition and condition. Nominal Pfa=0.01,
calibrated per acquisition. CCEBM-classical uses the appropriate polarimetric
observations and some conventional comparators use HH: report this input
configuration rather than implying all methods were given identical channels.

`python -m pytest tests -q` and `python repro_norival_du_ipix.py` regenerate
all three CSVs and five figure pairs using the *cleaned merged sweep JSON*.
The `ROC-style` figure depicts summary operating-point envelopes; it is not
an independently reconstructed ROC from saved individual window scores.

## Corrected headline median Pd @ nominal Pfa=0.01, M=128

| SCR (dB) | DU +21Hz | PU +21Hz | AMF +21Hz | DU +60Hz | PU +60Hz | AMF +60Hz |
|---:|---:|---:|---:|---:|---:|---:|
| -30 | 0.0100 | 0.0167 | 0.0067 | 0.0367 | 0.0200 | 0.0133 |
| -25 | 0.0200 | 0.0200 | 0.0067 | 0.0767 | 0.0567 | 0.0100 |
| -20 | 0.0800 | 0.1000 | 0.0067 | 0.2233 | 0.1667 | 0.0067 |
| -15 | 0.2333 | 0.2367 | 0.0067 | 0.4267 | 0.4233 | 0.0067 |
| -10 | 0.4500 | 0.4267 | 0.0133 | 0.5733 | 0.6033 | 0.0200 |
| -5 | 0.5900 | 0.6067 | 0.0167 | 0.7933 | 0.9433 | 0.0833 |
| +0 | 0.8000 | 0.8167 | 0.1500 | 0.9133 | 0.9867 | 0.4467 |
| +5 | 0.8867 | 0.9600 | 0.6533 | 0.9700 | 1.0000 | 0.9400 |

Numbers above are generated directly from `data_norival_DU_ipix/ipix_pd_median_norival_du.csv`, not prior README approximations. The sign-test evidence is in `ipix_effects_norival_du.csv`; interpret p values within the declared hypothesis family (16 offset-by-comparator tests). The nine evaluated arms are DU-CCEBM, CCEBM-PU/WL/DP, STFICA, Kelly GLRT, AMF, ANMF and CA-CFAR. CERBM-family numeric results have been removed from the copied sweep, but `code/cerbm.py` remains a classical helper dependency.

The bundled data contain saved sweep statistics, not raw I/Q radar measurements or individual H0/H1 score arrays. `code/repro_sweep.py`, `code/sweep_ipix_v4.py`, and `code/ipix_reader.py` provide provenance for the original run; raw-data collection requires official CDF files and the original environment. Code under BSD-2-Clause (see LICENSE); raw data terms are separate.
