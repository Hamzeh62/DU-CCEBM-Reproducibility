# Experimental evidence provenance — distinct model families

**Two-layer IPIX (root).** The original immutable-source 41-file manifest, prior-validation-frozen
weights, and scores cover the long-CPI IPIX M=512/1024 matched run. The historical comparator
files and long-CPI numeric summaries are retained; the original historical PDF figure set was
excluded from the clean public export. Some classical algorithms use different channel inputs,
and Pfa is a nominal calibrated threshold with additional holdout diagnostics.

**Four-layer NetRAD/IPIX (`extensions/netrad-ipix-4layer`).** Contains 21 NetRAD acquisition score
snapshots and eight IPIX acquisition summary records. All eight original numerical CSV products
can be regenerated from archived snapshots, without raw radar recordings. The four-layer
per-window optimizer does not inherit timing measurements from the frozen two-layer code.

**Six-layer IPIX (`extensions/ipix-six-layer-frozen`).** Recovered training/inference source and
real frozen `real_A_best_model.pt` checkpoint, 240 mixed and 22,416 genuine-only archived score
comparisons; all 22,656 scores were recomputed from 14 raw IPIX recordings within 1e-6. The
genuine-only set is not an entirely held-out acquisition; the full training process has not been
independently rerun. The obsolete manuscript draft and historical figures were excluded.

No raw radar CDF/IQ recordings are shipped. The `cerbm.py` helper files are preserved for legacy
CCEBM dependencies, with no CERBM-family numerical detector results included.
