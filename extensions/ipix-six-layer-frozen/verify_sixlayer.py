#!/usr/bin/env python3
"""Verify recovered six-layer frozen checkpoint and archived score evidence.

Modes: cached-data audit, and optionally full raw IPIX inference audit.
Never trains, publishes, or writes over any archived experimental result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
CHECKPOINT_SHA='5d450c073c22c3cb9aeca8b49fe837549f3921d347439b3db268f3234981fb01'
RAW_TOL=1e-6
EXPECTED_MIXED_AUC=0.5261752512121425
EXPECTED_CLEAN_AUC=0.8042931034482759

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for buf in iter(lambda:fh.read(1<<20),b''):h.update(buf)
    return h.hexdigest()

def validate_manifest():
    lines=(ROOT/'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    count=0
    for line in lines:
        h,name=line.split('  ',1)
        p=ROOT/name
        if not p.is_file():raise AssertionError('missing '+name)
        if sha256(p)!=h:raise AssertionError('SHA256 mismatch '+name)
        count+=1
    print(f'PASS manifest: {count} source payload files')

def load_checkpoint():
    p=ROOT/'checkpoints/real_A_best_model.pt'
    assert sha256(p)==CHECKPOINT_SHA,'checkpoint SHA256 differs from frozen original'
    ckpt=torch.load(p,map_location='cpu',weights_only=True)
    cfg=ckpt['cfg']
    assert ckpt['training_complete'] is True
    assert (cfg['L_net'],cfg['L_filt'],cfg['K_white'],cfg['T'])==(6,6,4,128)
    assert int(cfg['seed'])==42
    import du_ccebm_simulation as sim
    sim.set_seed(42)
    model=sim.DUCCEBM(L_net=cfg['L_net'],N=cfg['N'],L_filt=cfg['L_filt'],K_white=cfg['K_white'],**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model'],strict=True)
    model.eval()
    assert len(ckpt['model'])==100
    with torch.no_grad():
        x=torch.randn(3,2,2,128)
        scores,*_=model(x)
    assert scores.shape==(3,)
    assert bool(torch.isfinite(scores).all())
    assert bool(((scores>=0)&(scores<=1)).all())
    print('PASS checkpoint SHA256, six layers, strict 100-entry load and finite inference')
    return model,ckpt

def validate_cache():
    with np.load(ROOT/'outputs/revision_figures_scores.npz',allow_pickle=False) as z:
        assert z['mixed_labels'].shape==(240,)
        assert z['clean_labels'].shape==(22416,)
        mixed_labels=z['mixed_labels'].copy()
        clean_labels=z['clean_labels'].copy()
        mixed_du=z['mixed_du'].copy()
        clean_du=z['clean_du'].copy()
        mixed_auc=roc_auc_score(mixed_labels,mixed_du)
        clean_auc=roc_auc_score(clean_labels,clean_du)
        assert abs(mixed_auc-EXPECTED_MIXED_AUC)<2e-7
        assert abs(clean_auc-EXPECTED_CLEAN_AUC)<2e-7
    with np.load(ROOT/'results/revision_v2_scores.npz',allow_pickle=False) as v:
        assert v['labels'].shape==(22416,)
        assert np.array_equal(v['labels'],clean_labels)
        assert np.array_equal(v['du'],clean_du)
        assert len(set(v['file_ids'].tolist()))==14
    j=json.loads((ROOT/'outputs/revision_figures_audit.json').read_text())
    assert j['mixed_n']==240 and j['clean_n']==22416
    assert abs(j['mixed_auc']['DU-CCEBM']-mixed_auc)<2e-7
    assert abs(j['clean_auc']['DU-CCEBM']-clean_auc)<2e-7
    rev=json.loads((ROOT/'results/REVISION_REAL_RESULTS_v2.json').read_text())
    assert rev['pool']['n']==22416
    print(f'PASS archives: 240 mixed and 22416 clean DU scores; AUC={mixed_auc:.9f} and {clean_auc:.9f}')
    return {'mixed_labels':mixed_labels,'clean_labels':clean_labels,'mixed_du':mixed_du,'clean_du':clean_du}

def validate_raw(raw,model,cache):
    files=sorted(raw.glob('*.cdf'))
    if len(files)!=14:
        raise ValueError('Expected exactly the 14 original 1993 Dartmouth IPIX .cdf recordings, got '+str(len(files)))
    import du_ccebm_simulation as sim
    from scripts.reproduce_revision_figures import config_from_results,build_mixed_split,clean_genuine_pool,du_scores
    if not sim._NETCDF4_AVAILABLE:
        from scipy.io import netcdf_file
        from functools import partial
        sim.NCDataset=partial(netcdf_file,mmap=False)
        sim._NETCDF4_AVAILABLE=True
        print('NOTE: reading original NetCDF3 bytes via SciPy fallback')
    real=json.loads((ROOT/'REVISION_REAL_RESULTS.json').read_text())
    cfg=config_from_results(real)
    cfg['ipix_dir']=str(raw.resolve())
    full, _, _, x_mixed, y_mixed=build_mixed_split(cfg,real['real_A_train_du_ccebm']['n_samples_used_subset'],42)
    assert len(full)==45000,'unanticipated raw data sample count'
    assert np.array_equal(y_mixed.numpy().astype(np.int64),cache['mixed_labels']),'mixed labels mismatch'
    x_clean,y_clean,n_excluded=clean_genuine_pool(cfg,full,real)
    assert len(x_clean)==22416 and n_excluded==1134
    assert np.array_equal(y_clean.astype(np.int64),cache['clean_labels']),'clean labels mismatch'
    fresh_mixed=du_scores(model,x_mixed,batch=256)
    fresh_clean=du_scores(model,x_clean,batch=256)
    mixed_err=float(np.max(np.abs(fresh_mixed-cache['mixed_du'])))
    clean_err=float(np.max(np.abs(fresh_clean-cache['clean_du'])))
    assert mixed_err<=RAW_TOL and clean_err<=RAW_TOL, f'raw discrepancy mixed={mixed_err} clean={clean_err}'
    print(f'PASS RAW IPIX: 14 files, 45000 windows; mixed max error {mixed_err:.3g}; clean max error {clean_err:.3g}')
    print(f'PASS fresh raw AUC: mixed={roc_auc_score(cache["mixed_labels"],fresh_mixed):.9f}; clean={roc_auc_score(cache["clean_labels"],fresh_clean):.9f}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-ipix-dir',type=Path,default=None,help='Optional directory of exactly 14 original 1993 IPIX CDF files')
    args=ap.parse_args()
    validate_manifest()
    model,_=load_checkpoint()
    cache=validate_cache()
    if args.raw_ipix_dir:
        if not args.raw_ipix_dir.is_dir():ap.error('not a directory: '+str(args.raw_ipix_dir))
        validate_raw(args.raw_ipix_dir,model,cache)
    else:
        print('NOTE raw IPIX reread not requested; use --raw-ipix-dir to independently re-score from source CDF')
    print('PASS six-layer curated frozen-checkpoint verification')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('FAIL:',repr(exc),file=sys.stderr);sys.exit(1)
