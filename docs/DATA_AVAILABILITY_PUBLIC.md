# Data and code availability

Only derived numeric scores, aggregated comparisons and source code are included. No original
IPIX CDF or NetRAD raw I/Q files are redistributed. To recompute six-layer frozen inference from
signal bytes, separately obtain exactly the original 14 IPIX CDF recordings, then run
`extensions/ipix-six-layer-frozen/verify_sixlayer.py --raw-ipix-dir /path/to/data`.
For four-layer NetRAD, the included 21 JSON score snapshots can regenerate all five CSVs;
for four-layer IPIX, the included eight file summaries can regenerate all three CSVs.
The bundled four-layer release does not itself provide a raw-to-result NetRAD reproduction.
DOI will be supplied only after a published Zenodo record is verified.
