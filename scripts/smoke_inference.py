#!/usr/bin/env python3
"""Short deterministic same-input H0 smoke test for the two actual solvers."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import pathlib,sys,numpy as np,torch
from threadpoolctl import threadpool_limits
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/classical'),str(ROOT/'src/du')]
from ccebm_v3 import ccebm_v3
from du_ccebm_matched_development import DUCCEBM
rng=np.random.default_rng(7)
x=rng.normal(size=(2,5000+128))+1j*rng.normal(size=(2,5000+128))
with threadpool_limits(limits=1):
 torch.set_num_threads(1)
 y,info=ccebm_v3(x,L=6,K=4,n_iter=1,paraunitary_on=True,whiten_mode='spatial',n_restart=1,seed=7)
 model=DUCCEBM(L=6,K=4,depth=2,seed=7)
 ck=torch.load(ROOT/'checkpoints/DEPTH2_SELECTED_PRIOR_VALIDATION_ONLY.pt',weights_only=True,map_location='cpu')
 model.load_state_dict(ck['model_state_dict']);model.eval()
 score,aux=model(torch.as_tensor(x,dtype=torch.complex128),21.,M=128)
 assert np.isfinite(y).all() and np.isfinite(float(score.detach()))
 assert float(info['paraunitary_residual'])<1e-9
 assert float(aux['pu_residual'].detach())<1e-9
 print('PASS classical and DU finite; PU residual:',info['paraunitary_residual'],float(aux['pu_residual'].detach()))
