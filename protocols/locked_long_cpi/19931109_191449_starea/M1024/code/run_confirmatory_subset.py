"""Exact frozen v4-extension, resumable 5-part full H0 1500 + H1 150/SCR.

No retraining. Same injected samples for original CCEBM and DU. This is a
single-file/single-CPI/one-Doppler subset, NOT the full long-CPI grid.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'reference/original'),str(ROOT/'reference/du')]
from ccebm_v3 import ccebm_v3
from du_ccebm_matched_development import DUCCEBM
from ipix_reader import cell_slab, NEEDED, clutter_centroid_hz

FILE = '19931109_191449_starea.cdf'
M = 1024
OFFSET = 60.0
SCRS = (-25, -20)
N_H0 = 1500
N_H1 = 150
TAIL = 5000
SEED = 7
POL = np.array([1, np.exp(1j*np.pi/4)], dtype=np.complex128)/np.sqrt(2)
CK = ROOT/'reference/du/DEPTH2_SELECTED_PRIOR_VALIDATION_ONLY.pt'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def atomic(path, ob):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(ob,indent=2,allow_nan=False))
    tmp.replace(path)

def run(worker, workers, smoke=False):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    p=Path(os.environ.get('IPIX_DATA_ROOT','/mnt/data'))/FILE
    slab,_=cell_slab(str(p), str(ROOT/'cache'))
    x=slab[:,NEEDED.index(0),:].astype(np.complex128).T
    assert x.shape[0]==2
    centroid=clutter_centroid_hz(x[0]); freq=(centroid+OFFSET+500)%1000-500
    power=float(np.mean(np.abs(x[0])**2))
    start_h0=np.linspace(TAIL,x.shape[1]-1024-1,N_H0).astype(int)
    start_h1=np.linspace(TAIL,x.shape[1]-1024-1,N_H1).astype(int)
    assert len(set(start_h0))==N_H0 and len(set(start_h1))==N_H1
    jobs=[('H0',None,i,int(s)) for i,s in enumerate(start_h0)]
    jobs += [('H1',s,i,int(st)) for s in SCRS for i,st in enumerate(start_h1)]
    assert len(jobs)==1800
    model=DUCCEBM(L=6,K=4,depth=2,seed=SEED)
    model.load_state_dict(torch.load(CK,map_location='cpu',weights_only=True)['model_state_dict'])
    model.eval()
    meta={'file':FILE,'M':M,'doppler_offset':OFFSET,'target_doppler_hz':float(freq),
      'scrs':list(SCRS),'h0_n':N_H0,'h1_n_per_scr':N_H1,'tail':TAIL,'classic_iters':15,
      'classic_restarts':3,'du_depth':2,'L':6,'K':4,'seed':SEED,
      'target_pol':'[1,exp(j*pi/4)]/sqrt(2)', 'target_injection':'full_history_plus_M',
      'data_sha256':sha(p),'ccebm_sha256':sha(ROOT/'reference/original/ccebm_v3.py'),
      'checkpoint_sha256':sha(CK), 'workers':workers,'worker':worker}
    out=ROOT/'results'/('SMOKE_'+str(worker)+'.json' if smoke else 'part_'+str(worker)+'_of_'+str(workers)+'.json')
    st=json.loads(out.read_text()) if out.exists() else {'meta':meta,'rows':[]}
    assert st['meta']==meta
    done={r['job_index'] for r in st['rows']}
    jobs_for_worker=[(j,*case) for j,case in enumerate(jobs) if j%workers==worker]
    if smoke: jobs_for_worker=jobs_for_worker[:2]
    with threadpool_limits(limits=1):
        for j,kind,scr,i,start in jobs_for_worker:
            if j in done: continue
            z=x[:,start-TAIL:start+M].copy()
            assert z.shape==(2,TAIL+M)
            if scr is not None:
                ampl=np.sqrt(power*10**(scr/10))
                t=np.arange(TAIL+M)
                z+=POL[:,None]*(ampl*np.exp(2j*np.pi*freq*t/1000))[None,:]
            tt=time.perf_counter()
            y,info=ccebm_v3(z,L=6,K=4,n_iter=15,paraunitary_on=True,
                              whiten_mode='spatial',n_restart=3,seed=SEED)
            v=np.exp(-2j*np.pi*freq*np.arange(M)/1000)
            cc=float(np.max(np.abs(y[:,-M:]@v)**2/(np.sum(np.abs(y[:,-M:])**2,axis=1)+1e-30)))
            cc_ms=(time.perf_counter()-tt)*1000
            tt=time.perf_counter()
            du,aux=model(torch.as_tensor(z,dtype=torch.complex128),freq,M=M)
            du_ms=(time.perf_counter()-tt)*1000
            row={'job_index':j,'kind':kind,'scr':scr,'i':i,'start':start,
                 'cc_score':cc,'du_score':float(du.detach()),
                 'cc_ms':cc_ms,'du_ms':du_ms,'cc_pu':float(info['paraunitary_residual']),
                 'du_pu':float(aux['pu_residual'].detach())}
            assert np.all(np.isfinite([row[k] for k in ('cc_score','du_score','cc_ms','du_ms','cc_pu','du_pu')]))
            assert max(row['cc_pu'],row['du_pu'])<1e-9
            st['rows'].append(row)
            if len(st['rows'])%10==0:
                atomic(out,st)
                print('PROGRESS',worker,len(st['rows']),'/',len(jobs_for_worker),flush=True)
    atomic(out,st)
    print('DONE',worker,len(st['rows']),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker',type=int,required=True)
    ap.add_argument('--workers',type=int,default=5)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    assert 0<=a.worker<a.workers
    run(a.worker,a.workers,a.smoke)
