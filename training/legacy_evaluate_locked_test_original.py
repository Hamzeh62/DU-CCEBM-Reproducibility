"""TEST-only locked DU prototype evaluation, same raw IPIX positions and fixed score as legacy v4.
No model selection or optimizer calls; calibration H0 and H1 paired with archived classical CCEBM.
Empirical calibration Pfa at training H0 is *not* evidence of out-of-sample CFAR.
"""
from __future__ import annotations
import argparse,sys,json,hashlib,math,os,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/data/DU_CCEBM_Next_Robustness_2026-10-08')
sys.path[:0]=[str(SOURCE/'reference'),str(SOURCE/'development_only')]
from ipix_reader import cell_slab,NEEDED,PRF_HZ,clutter_centroid_hz
from du_ccebm_matched_development import DUCCEBM,lag_stack,complex_spatial_whiten,lattice_bank,entropy_cost,full_fir_residual
MS=(10,32,128); OFFSETS=(21.,60.); SCR=(-30,-25,-20,-15,-10,-5,0,5)

def atomic_json(path,dat):
 tmp=str(path)+'.tmp';Path(tmp).write_text(json.dumps(dat,indent=1));os.replace(tmp,path)

def separated(model,z):
 # Exact same sequence as model.forward, except scores can be read for multiple M and Dopplers.
 Zw=lag_stack(complex_spatial_whiten(z.to(torch.complex128)),model.L)
 U=torch.linalg.qr(model.U0.conj().T).Q.conj().T
 vs=[v/(torch.linalg.vector_norm(v)+1e-30) for v in model.V0]
 for k in range(model.depth):
  W=lattice_bank(U,vs)
  with torch.enable_grad():
   c=entropy_cost(W,Zw,model.K)
   g=torch.autograd.grad(c,[U]+vs,create_graph=model.training)
  step=torch.clamp(F.softplus(model.raw_steps[k]),min=1e-7,max=.02)
  U=torch.linalg.qr((U-step*g[0]).conj().T).Q.conj().T
  vs=[(v-step*gv)/(torch.linalg.vector_norm(v-step*gv)+1e-30) for v,gv in zip(vs,g[1:])]
 W=lattice_bank(U,vs)
 y=W.reshape(2,-1).conj()@Zw
 residual=float(full_fir_residual(W).detach())
 return y.detach(),residual

def scores(y,f):
 out={}
 for m in MS:
  yy=y[:,-m:]
  n=torch.arange(m,dtype=torch.float64)
  v=torch.exp((-2j*math.pi*f/PRF_HZ)*n)
  a=torch.abs(torch.sum(yy*v,dim=-1))**2/(torch.sum(torch.abs(yy)**2,dim=-1)+1e-30)
  out[f'M{m}']=float(torch.max(a.real))
 return out

def single_test(model,x,f):
 a,r=separated(model,x)
 d=scores(a,f)
 for m in MS:
  v,_=model(x,f,M=m)
  diff=abs(d[f'M{m}']-float(v.detach()))
  assert diff < 1e-9,f'divergence M{m}: {diff}'
 return r,d

def add_tone(x,p,f,scr):
 t=torch.arange(x.shape[-1],dtype=torch.float64)
 pol=torch.tensor([1.,np.exp(1j*np.pi/4)],dtype=torch.complex128)/math.sqrt(2)
 amp=math.sqrt(p*10**(scr/10))
 return x+amp*pol[:,None]*torch.exp((2j*math.pi*f/PRF_HZ)*t)[None,:]

def dkey(d):return 'd%d'%int(d)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,choices=(0,1),required=True);args=parser.parse_args()
 torch.set_num_threads(1)
 with threadpool_limits(limits=1):
  lock=json.loads((ROOT/'LOCKED_MODEL_SELECTION.json').read_text())
  assert lock['selection_locked'] and lock['test_not_used']
  ckpath=ROOT/'SELECTED_VALIDATION_ONLY.pt'
  assert hashlib.sha256(ckpath.read_bytes()).hexdigest()==lock['checkpoint_sha256']
  ck=torch.load(ckpath,map_location='cpu',weights_only=True)
  assert ck['epoch']==lock['selected_epoch']
  model=DUCCEBM(L=6,K=4,depth=2,seed=7)
  model.load_state_dict(ck['model_state_dict']);model.eval()
  name=['19931111_163625_starea.cdf','19931118_162658_stareC0000.cdf'][args.index]
  assert name in json.loads((ROOT/'FROZEN_TRAIN_VALIDATION_PROTOCOL.json').read_text())['test_UNREAD']
  path=Path('/mnt/data')/name
  slab,meta=cell_slab(str(path),str(ROOT/'cache'))
  x=slab[:,NEEDED.index(0),:].astype(np.complex128).T
  x=torch.tensor(x.copy(),dtype=torch.complex128)
  cent=clutter_centroid_hz(x[0].numpy()); p=float(np.mean(np.abs(x[0].numpy())**2))
  fd={dkey(d):float((cent+d+PRF_HZ/2)%PRF_HZ-PRF_HZ/2) for d in OFFSETS}
  with zipfile.ZipFile('/mnt/data/My_Result.zip') as z:
   hist=json.loads(z.read('results/ipix_sweep_v4.json'))['files'][name]
  for dk in fd:
   assert abs(fd[dk]-hist['target_doppler_hz'][dk])<1e-9
  s0=np.linspace(5000,x.shape[1]-128-1,1500).astype(int)
  s1=np.linspace(5000,x.shape[1]-128-1,150).astype(int)
  stem=path.stem;chk=ROOT/f'{stem}_DU_TEST_ONLY.json';t0=time.time()
  if chk.exists():
   state=json.loads(chk.read_text())
   assert state['model_sha256']==lock['checkpoint_sha256']
  else:
   state={'file':name,'model_sha256':lock['checkpoint_sha256'],'selected_epoch':lock['selected_epoch'],'cdf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'INCOMPLETE','centroid_hz':cent,'target_doppler_hz':fd,'n_h0_complete':0,'H0':{dkey(d):{f'M{m}':[] for m in MS} for d in OFFSETS},'threshold':{},'levels':{},'h1_counts':{},'max_paraunitary_residual':0.,'protocol':'original v4 positions; 1500 H0 and 150 H1 x 8 SCR x 2 Dopplers; threshold empirical quantile 0.99; same targets and score family; DU prototype depth2 vs legacy CCEBM 15 iter/3 restarts'}
  # Numerical identity check of fused M readout with original model.forward.
  zz=x[:,s0[0]-5000:s0[0]+128]
  r,_=single_test(model,zz,fd['d21'])
  assert r<1e-9
  for i in range(state['n_h0_complete'],len(s0)):
   s=int(s0[i]);y,r=separated(model,x[:,s-5000:s+128])
   state['max_paraunitary_residual']=max(state['max_paraunitary_residual'],r)
   for dk,fi in fd.items():
    val=scores(y,fi)
    for mm in val:state['H0'][dk][mm].append(val[mm])
   state['n_h0_complete']=i+1
   if (i+1)%250==0:
    atomic_json(chk,state)
    print('H0',name,i+1,'/',len(s0),f'{time.time()-t0:.1f}s',flush=True)
  if not state['threshold']:
   state['threshold']={dk:{mm:float(np.quantile(v,.99)) for mm,v in series.items()} for dk,series in state['H0'].items()}
   atomic_json(chk,state)
   print('H0 CALIBRATED',name,flush=True)
  for dlt in OFFSETS:
   dk=dkey(dlt);fi=fd[dk]
   for scr in SCR:
    case=f'{dk}|{scr}'
    if case in state['levels']:continue
    detected={f'M{m}':0 for m in MS}
    accum={f'M{m}':[] for m in MS}
    for s in s1:
     s=int(s);chunk=x[:,s-5000:s+128]
     injection=add_tone(chunk,p,fi,scr)
     sep,r=separated(model,injection)
     state['max_paraunitary_residual']=max(state['max_paraunitary_residual'],r)
     for mm,v in scores(sep,fi).items():
      accum[mm].append(v)
      detected[mm]+=int(v>state['threshold'][dk][mm])
    state['levels'][case]={mm:float(detected[mm]/len(s1)) for mm in detected}
    state['h1_counts'][case]=detected
    state.setdefault('H1_scores',{})[case]=accum
    atomic_json(chk,state)
    print('H1',name,case,'/',len(SCR)*len(OFFSETS),'counts',detected,'elapsed',f'{time.time()-t0:.1f}s',flush=True)
  state['status']='COMPLETE' if len(state['levels'])==16 else 'INCOMPLETE'
  state['elapsed_seconds_this_session']=round(time.time()-t0,1)
  state['historic_CCEBM_from_My_Result_not_rerun_now']={case:{f'M{m}':hist['levels'][case][f'ccebm_pu_M{m}'] for m in MS} for case in state['levels']}
  state['H0_time_split_diagnostic']={dk:{mm:{'threshold_first_1000':float(np.quantile(arr[:1000],.99)),'empirical_pfa_last_500':float(np.mean(np.array(arr[1000:])>np.quantile(arr[:1000],.99)))} for mm,arr in series.items()} for dk,series in state['H0'].items()}
  atomic_json(chk,state)
  print('TEST_DONE',name,state['status'],'seconds',state['elapsed_seconds_this_session'],flush=True)
if __name__=='__main__':main()
