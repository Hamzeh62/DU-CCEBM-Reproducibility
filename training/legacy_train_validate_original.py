"""PREDECLARED exploratory DU-CCEBM continuation; TRAIN/VAL ONLY.
Inputs genuinely extracted from raw IPIX; tests never opened by this script.
This is not classical CCEBM solver equivalence: only comparable input and readout.
"""
import os,sys,json,hashlib,time,math,random,shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/mnt/data/DU_CCEBM_Next_Robustness_2026-10-08')
sys.path[:0]=[str(SOURCE/'reference'),str(SOURCE/'development_only')]
from ipix_reader import cell_slab,NEEDED,PRF_HZ,clutter_centroid_hz
from du_ccebm_matched_development import DUCCEBM
FROZEN={
 'purpose':'Continuation of TRAIN-only engineering model, then validation-only model selection; no TEST access',
 'dataset_root':'/mnt/data',
 'train':['19931107_135603_starea.cdf','19931107_145028_starea.cdf','19931108_220902_starea.cdf','19931109_202217_starea.cdf'],
 'validation':['19931118_035737_stareC0000.cdf','19980204_163113_ANTSTEP.CDF'],
 'test_UNREAD':['19931111_163625_starea.cdf','19931118_162658_stareC0000.cdf'],
 'train_starts':[9000,21000,33000,45000,57000,69000],
 'validation_starts':[7000,11500,16000,20500,25000,29500,34000,38500,43000,47500],
 'scr_db':[-30,-20,-10], 'delta_hz':[21.,60.],
 'epochs_after_preliminary_checkpoint':4,'candidate_epochs':[0,1,2,3,4],
 'optimizer':'Adam','learning_rate':5e-5, 'train_loss':'softplus((0.75+score_H0-score_H1)/5), paired ranking',
 'validation_metric':'mean within-acquisition H0/H1 AUC, 2 acquisitions x 2 offsets x 3 SCR, tie choose earliest epoch',
 'model':{'L':6,'K':4,'depth':2,'seed':7,'score':'max two-source normalized coherence M=128','classical_restarts':3,'classical_iter':15},
 'human_policy':'test not loaded or used during checkpoint selection; exclude CERBM and DU-CERBM outputs'
}

def atomic_json(p,obj):
 t=str(p)+'.tmp';Path(t).write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(t,p)

def load_data(names):
 out={};meta={}
 for name in names:
  path=ROOT.parent/name
  if not path.exists():path=Path('/mnt/data')/name
  slab,_=cell_slab(str(path),str(ROOT/'cache'))
  x=slab[:,NEEDED.index(0),:].astype(np.complex128).T
  out[name]=(torch.tensor(x.copy(),dtype=torch.complex128),clutter_centroid_hz(x[0]),float(np.mean(np.abs(x[0])**2)))
  meta[name]={'cdf_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'n_sweeps':int(x.shape[1])}
  print('LOADED',name,x.shape,flush=True)
 return out,meta

def scene(x,centroid,p,s,scr,dlt):
 f=(centroid+dlt+PRF_HZ/2)%PRF_HZ-PRF_HZ/2
 z=x[:,s-5000:s+128].clone()
 if z.shape!=(2,5128):raise AssertionError(f'invalid {z.shape}, {s}')
 if scr is not None:
  t=torch.arange(z.shape[1],dtype=torch.float64)
  tgt=(p*10**(scr/10))**.5*torch.tensor([1.,np.exp(1j*np.pi/4)],dtype=torch.complex128)[:,None]/math.sqrt(2)*torch.exp((2j*math.pi*f/PRF_HZ)*t)[None,:]
  z=z+tgt
 return z,float(f)

def auc(h0,h1):
 a=np.array(h0)[:,None]; b=np.array(h1)[None,:]
 return float(np.mean(b>a)+.5*np.mean(b==a))

def validate(model,vals):
 model.eval();out=[];mx=0
 for nm,(x,centroid,p) in vals.items():
  for dlt in FROZEN['delta_hz']:
   h0=[];pos={scr:[] for scr in FROZEN['scr_db']}
   for s in FROZEN['validation_starts']:
    z,f=scene(x,centroid,p,s,None,dlt)
    a,aux=model(z,f,M=128);h0.append(float(a.detach()));mx=max(mx,float(aux['pu_residual'].detach()))
    for scr in FROZEN['scr_db']:
     zh,fh=scene(x,centroid,p,s,scr,dlt)
     b,aux=model(zh,fh,M=128);pos[scr].append(float(b.detach()));mx=max(mx,float(aux['pu_residual'].detach()))
   for scr in FROZEN['scr_db']:
    out.append({'file':nm,'doppler_offset':dlt,'scr':scr,'auc':auc(h0,pos[scr]),'n_h0':len(h0),'n_h1':len(pos[scr]),'H0_scores':h0,'H1_scores':pos[scr]})
 return float(np.mean([r['auc'] for r in out])),out,mx

def main():
 torch.set_num_threads(1);torch.manual_seed(7);np.random.seed(7);random.seed(7)
 with threadpool_limits(limits=1):
  atomic_json(ROOT/'FROZEN_TRAIN_VALIDATION_PROTOCOL.json',FROZEN)
  data,meta=load_data(FROZEN['train']+FROZEN['validation'])
  assert set(FROZEN['test_UNREAD']).isdisjoint(data)
  train={k:data[k] for k in FROZEN['train']};val={k:data[k] for k in FROZEN['validation']}
  model=DUCCEBM(L=6,K=4,depth=2,seed=7)
  seed=SOURCE/'development_only'/'TRAIN_ONLY_MATCHED_DEVELOPMENT.pt'
  ck=torch.load(seed,weights_only=True,map_location='cpu')
  model.load_state_dict(ck['model_state_dict'])
  source_hash=hashlib.sha256(seed.read_bytes()).hexdigest()
  optimizer=torch.optim.Adam([model.U0,*model.V0,model.raw_steps],lr=FROZEN['learning_rate'])
  records=[]; t0=time.time()
  for ep in FROZEN['candidate_epochs']:
   if ep>0:
    model.train();losses=[];maxpu=0
    # Stable deterministic task order across epochs.
    for nm,(x,centroid,p) in train.items():
     for s in FROZEN['train_starts']:
      for dlt in FROZEN['delta_hz']:
       clean,f=scene(x,centroid,p,s,None,dlt)
       for scr in FROZEN['scr_db']:
        tgt,ft=scene(x,centroid,p,s,scr,dlt)
        optimizer.zero_grad(set_to_none=True)
        score0,aux0=model(clean,f,M=128)
        score1,aux1=model(tgt,ft,M=128)
        loss=F.softplus((.75+score0-score1)/5)
        if not bool(torch.isfinite(loss)):raise AssertionError('nonfinite loss')
        loss.backward()
        for v in [model.U0,*model.V0,model.raw_steps]:
         if v.grad is None or not bool(torch.isfinite(v.grad).all()):raise AssertionError('invalid gradient')
        torch.nn.utils.clip_grad_norm_([model.U0,*model.V0,model.raw_steps],1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        maxpu=max(maxpu,float(aux0['pu_residual'].detach()),float(aux1['pu_residual'].detach()))
    trainloss=float(np.mean(losses))
   else: trainloss=None;maxpu=None
   metric,v_rows,vpu=validate(model,val)
   pt=ROOT/f'candidate_epoch_{ep}.pt'
   torch.save({'epoch':ep,'model_state_dict':model.state_dict(),'selection_metric_val_auc':metric,'architecture':'DUCCEBM depth2 prototype, NOT classical equivalence','source_preliminary_sha256':source_hash},pt)
   row={'epoch':ep,'train_pair_loss':trainloss,'validation_mean_auc':metric,'validation_max_paraunitary_residual':vpu,'train_max_paraunitary_residual':maxpu,'checkpoint':pt.name,'elapsed_sec':round(time.time()-t0,2)}
   records.append(row)
   atomic_json(ROOT/'TRAIN_VALIDATION_PROGRESS.json',{'protocol':FROZEN,'source_preliminary_sha256':source_hash,'train_cdf_hashes':{k:meta[k] for k in train},'validation_cdf_hashes':{k:meta[k] for k in val},'candidates':records,'selected':None,'test_evaluation':'NOT_STARTED'})
   (ROOT/f'validation_epoch_{ep}.json').write_text(json.dumps({'aggregate_auc':metric,'groups':v_rows},indent=1))
   print('EPOCH',json.dumps(row),flush=True)
  best=max(records,key=lambda r:(r['validation_mean_auc'],-r['epoch']))
  shutil.copy2(ROOT/best['checkpoint'],ROOT/'SELECTED_VALIDATION_ONLY.pt')
  selected={'selection_locked':True,'selected_epoch':best['epoch'],'validation_mean_auc':best['validation_mean_auc'],'selection_metric':'mean per-acquisition per-Doppler per-SCR AUC, held-out Validation only','preliminary_checkpoint_sha256':source_hash,'checkpoint_sha256':hashlib.sha256((ROOT/'SELECTED_VALIDATION_ONLY.pt').read_bytes()).hexdigest(),'candidate_history':records,'train_cdf_hashes':{k:meta[k] for k in train},'validation_cdf_hashes':{k:meta[k] for k in val},'test_not_used':True,'limits':'development prototype: depth2 learned steps/initialization vs classical 15 updates and 3 restarts; supervised pairwise ranking; nonblind historical test'}
  atomic_json(ROOT/'LOCKED_MODEL_SELECTION.json',selected)
  print('LOCKED_SELECTION',json.dumps({'epoch':best['epoch'],'val_auc':best['validation_mean_auc']}),flush=True)
if __name__=='__main__':main()
