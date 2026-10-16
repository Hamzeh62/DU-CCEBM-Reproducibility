# Data and code availability

Only derived numeric scores, aggregated comparisons and source code are included. No original
IPIX CDF or NetRAD raw I/Q files are redistributed. To recompute six-layer frozen inference from
signal bytes, separately obtain exactly the original 14 IPIX CDF recordings, then run
`extensions/ipix-six-layer-frozen/verify_sixlayer.py --raw-ipix-dir /path/to/data`.
For four-layer NetRAD, the included 21 JSON score snapshots can regenerate all five CSVs;
for four-layer IPIX, the included eight file summaries can regenerate all three CSVs.
The bundled four-layer release does not itself provide a raw-to-result NetRAD reproduction.
The exact published version 1.0.0 of this software is archived at https://doi.org/10.5281/zenodo.23416237.
The public code/data archive contains derived results and trained model checkpoints, not original raw
IPIX/NetRAD radar recordings; reproduce at the documented dataset and experiment scopes.
