#!/usr/bin/env python3
"""Independent metrics from frozen paired decision scores (no model run)."""
import argparse,csv,json,pathlib,collections,statistics
import numpy as np
from scipy.stats import rankdata
R=pathlib.Path(__file__).resolve().parents[1]

def auc(zero,one):
 n0,n1=len(zero),len(one); ranks=rankdata(np.r_[zero,one],method='average')
 return float((ranks[n0:].sum()-n1*(n1+1)/2)/(n0*n1))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='outputs/longcpi');a=ap.parse_args()
 target=R/a.out;target.mkdir(parents=True,exist_ok=True)
 per=[];temporal=[]; all_rows=[]
 for record in ['19931108_213827_starea','19931109_191449_starea']:
  for M in (512,1024):
   fn=R/'evidence/long_cpi'/f'paired_scores_{record}_M{M}.csv'
   with fn.open(newline='') as f:rows=list(csv.DictReader(f))
   assert len(rows)==1800,(fn,len(rows))
   ids=[int(x['job_index']) for x in rows];assert len(set(ids))==1800
   assert sorted(ids)==list(range(1800))
   h0=sorted([r for r in rows if r['kind']=='H0'],key=lambda x:int(x['job_index']))
   assert len(h0)==1500
   for scr in (-25,-20):
    h1=[r for r in rows if r['kind']=='H1' and int(float(r['scr']))==scr]
    assert len(h1)==150
    result={'recording':record,'M':M,'SCR_dB':scr,'n_H0':len(h0),'n_H1':len(h1)}
    for key in ('cc','du'):
     z=np.array([float(q[f'{key}_score']) for q in h0]);o=np.array([float(q[f'{key}_score']) for q in h1]);thr=np.quantile(z,.99,method='higher')
     result[f'{key}_threshold']=float(thr)
     result[f'{key}_Pd_nominal_0p01']=float(np.mean(o>thr))
     result[f'{key}_AUC']=auc(z,o)
    per.append(result)
   for key in ('cc','du'):
    z=np.array([float(q[f'{key}_score']) for q in h0]);thr=np.quantile(z[:1000],.99,method='higher')
    temporal.append({'recording':record,'M':M,'method':key,'H0_cal':1000,'H0_test':500,'Pfa_temporal_holdout':float(np.mean(z[1000:]>thr))})
   all_rows.extend(rows)
 agg=[]
 for M in (512,1024):
  for scr in (-25,-20):
   rr=[r for r in per if r['M']==M and r['SCR_dB']==scr]
   assert len(rr)==2
   agg.append({'M':M,'SCR_dB':scr,'num_acquisitions':2,
      'cc_mean_Pd':statistics.mean(r['cc_Pd_nominal_0p01'] for r in rr),
      'du_mean_Pd':statistics.mean(r['du_Pd_nominal_0p01'] for r in rr),
      'delta':statistics.mean(r['du_Pd_nominal_0p01']-r['cc_Pd_nominal_0p01'] for r in rr),
      'cc_mean_AUC':statistics.mean(r['cc_AUC'] for r in rr),
      'du_mean_AUC':statistics.mean(r['du_AUC'] for r in rr)})
 with (R/'evidence/long_cpi/TWO_ACQUISITION_AGGREGATE.csv').open(newline='') as f:
  archived={(int(r['M']),int(r['SCR_dB'])):r for r in csv.DictReader(f)}
 for x in agg:
  orig=archived[(x['M'],x['SCR_dB'])]
  for key in ['cc_mean_Pd','du_mean_Pd','delta','cc_mean_AUC','du_mean_AUC']:
   assert abs(x[key]-float(orig[key]))<5e-11,(x,key,orig[key])
 def write_csv(rel,rows):
  with (target/rel).open('w',newline='') as f:
   wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 write_csv('per_acquisition_metrics.csv',per);write_csv('aggregated_metrics.csv',agg)
 write_csv('temporal_h0_holdout.csv',temporal)
 print('PASS',len(all_rows),'paired windows, 4 acquisition-CPI combinations, 8 H1 conditions, all archived aggregates match')
 for row in agg:print(row)
 if any(float(q['Pfa_temporal_holdout'])>0.01 for q in temporal):
  print('NOTICE: nominal Pfa=0.01 is not guaranteed on temporal holdout')
if __name__=='__main__':main()
