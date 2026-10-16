#!/usr/bin/env python3
"""Frozen long-CPI core, configurable files, atomic resume; does not retrain DU."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,hashlib,json,pathlib,sys,time
import numpy as np,torch
from threadpoolctl import threadpool_limits
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/classical'),str(ROOT/'src/du'),str(ROOT/'src/ipix')]
from ccebm_v3 import ccebm_v3
from du_ccebm_matched_development import DUCCEBM
from ipix_reader import cell_slab,NEEDED,clutter_centroid_hz
RECORDINGS=('19931108_213827_starea.cdf','19931109_191449_starea.cdf')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--data-dir',required=True,type=pathlib.Path)
 ap.add_argument('--recording',choices=RECORDINGS,required=True)
 ap.add_argument('--M',type=int,choices=(512,1024),required=True)
 ap.add_argument('--out-dir',type=pathlib.Path,default=ROOT/'outputs'/'run')
 ap.add_argument('--worker',type=int,default=0)
 ap.add_argument('--workers',type=int,default=1)
 ap.add_argument('--smoke',action='store_true')
 a=ap.parse_args();assert 0<=a.worker<a.workers
 a.out_dir.mkdir(parents=True,exist_ok=True)
 data=a.data_dir/a.recording
 if not data.is_file():raise SystemExit(f'MISSING raw recording {data}')
 torch.set_num_threads(1);torch.manual_seed(7)
 slab,_=cell_slab(str(data),str(a.out_dir/'cache'))
 x=slab[:,NEEDED.index(0),:].astype(np.complex128).T
 assert x.shape[0]==2
 freq=float((clutter_centroid_hz(x[0])+60.0+500)%1000-500)
 power=float(np.mean(np.abs(x[0])**2))
 TAIL=5000; M=a.M
 # Replicate the *original frozen generator*: fixed -1024 margin even for M=512.
 starts0=np.linspace(TAIL,x.shape[1]-1024-1,1500).astype(int)
 starts1=np.linspace(TAIL,x.shape[1]-1024-1,150).astype(int)
 jobs=[('H0',None,i,int(s)) for i,s in enumerate(starts0)]
 jobs += [('H1',scr,i,int(s)) for scr in (-25,-20) for i,s in enumerate(starts1)]
 selected=[(j,*job) for j,job in enumerate(jobs) if j%a.workers==a.worker]
 if a.smoke:selected=selected[:3]
 ck=ROOT/'checkpoints/DEPTH2_SELECTED_PRIOR_VALIDATION_ONLY.pt'
 model=DUCCEBM(L=6,K=4,depth=2,seed=7)
 model.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model_state_dict']);model.eval()
 out=a.out_dir/('SMOKE' if a.smoke else 'full')
 out=out.with_name(out.name+f'_{a.worker}_of_{a.workers}.json')
 meta={'recording':a.recording,'M':M,'offset_hz':60,'worker':a.worker,'workers':a.workers,'data_sha256':sha(data),'checkpoint_sha256':sha(ck),'note':'observed injected-SCR protocol; not proof of CFAR'}
 if out.exists():
  state=json.loads(out.read_text());assert state['meta']==meta
 else:state={'meta':meta,'rows':[]}
 done={x['job_index'] for x in state['rows']}
 POL=np.array([1,np.exp(1j*np.pi/4)],dtype=np.complex128)/np.sqrt(2)
 with threadpool_limits(limits=1):
  for j,kind,scr,i,start in selected:
   if j in done:continue
   z=x[:,start-TAIL:start+M].copy();assert z.shape==(2,TAIL+M)
   if scr is not None:
    amp=np.sqrt(power*10**(scr/10));t=np.arange(TAIL+M)
    z+=POL[:,None]*(amp*np.exp(2j*np.pi*freq*t/1000))[None,:]
   t0=time.perf_counter()
   y,info=ccebm_v3(z,L=6,K=4,n_iter=15,paraunitary_on=True,whiten_mode='spatial',n_restart=3,seed=7)
   v=np.exp(-2j*np.pi*freq*np.arange(M)/1000)
   score=float(np.max(np.abs(y[:,-M:]@v)**2/(np.sum(np.abs(y[:,-M:])**2,axis=1)+1e-30)))
   cc_ms=(time.perf_counter()-t0)*1000
   t0=time.perf_counter();q,aux=model(torch.as_tensor(z,dtype=torch.complex128),freq,M=M)
   du_ms=(time.perf_counter()-t0)*1000
   row={'job_index':j,'kind':kind,'scr':scr,'i':i,'start':start,'cc_score':score,'du_score':float(q.detach()),'cc_ms':cc_ms,'du_ms':du_ms,'cc_pu':float(info['paraunitary_residual']),'du_pu':float(aux['pu_residual'].detach())}
   assert all(np.isfinite(row[k]) for k in ('cc_score','du_score','cc_ms','du_ms','cc_pu','du_pu'))
   assert max(row['cc_pu'],row['du_pu'])<1e-9
   state['rows'].append(row)
   if len(state['rows'])%10==0:
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(state,allow_nan=False));temp.replace(out)
  temp=out.with_suffix('.tmp');temp.write_text(json.dumps(state,allow_nan=False));temp.replace(out)
 print('PASS',len(state['rows']), 'entries; file:',out)
if __name__=='__main__':main()
