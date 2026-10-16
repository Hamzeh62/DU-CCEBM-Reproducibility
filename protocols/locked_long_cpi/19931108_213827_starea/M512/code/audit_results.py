"""Strict offline audit of the locked paired score subset. No model execution."""
from __future__ import annotations
import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
RS=ROOT/'results'
SCRS=[-25,-20]

def auc(s0,s1):
    s=np.r_[s0,s1]
    r=rankdata(s,method='average')
    n0,n1=len(s0),len(s1)
    return float((r[n0:].sum() - n1*(n1+1)/2)/(n0*n1))

def emit_csv(name,rows):
    assert rows
    with (RS/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def main():
    p=[RS/f'part_{i}_of_5.json' for i in range(5)]
    if any(not q.exists() for q in p): raise SystemExit('INCOMPLETE: worker files missing')
    parts=[json.loads(q.read_text()) for q in p]
    rows=[r for part in parts for r in part['rows']]
    assert len(rows)==1800, f'INCOMPLETE: {len(rows)} / 1800'
    ix=[r['job_index'] for r in rows]
    assert len(set(ix))==1800 and set(ix)==set(range(1800))
    rows=sorted(rows,key=lambda r:r['job_index'])
    assert len([r for r in rows if r['kind']=='H0'])==1500
    assert all(len([r for r in rows if r['kind']=='H1' and r['scr']==s])==150 for s in SCRS)
    assert all(max(r['cc_pu'],r['du_pu'])<1e-9 for r in rows)
    emit_csv('ALL_1800_PAIRED_SCORES.csv',rows)
    h0=[r for r in rows if r['kind']=='H0']
    summaries=[];cal=[]
    for scr in SCRS:
        h1=[r for r in rows if r['kind']=='H1' and r['scr']==scr]
        d={'SCR_dB':scr,'H0_N':len(h0),'H1_N':len(h1)}
        for m in ('cc','du'):
            zero=np.array([r[m+'_score'] for r in h0]); one=np.array([r[m+'_score'] for r in h1])
            thr=float(np.quantile(zero,.99,method='higher'))
            pred=one>thr
            d[m+'_threshold_0p01']=thr
            d[m+'_Pd_nominal_0p01']=float(pred.mean())
            d[m+'_detections']=int(pred.sum())
            d[m+'_AUC_all_H0']=auc(zero,one)
        d['delta_Pd_DU_minus_classic']=d['du_Pd_nominal_0p01']-d['cc_Pd_nominal_0p01']
        cc_pred=np.array([r['cc_score']>d['cc_threshold_0p01'] for r in h1])
        du_pred=np.array([r['du_score']>d['du_threshold_0p01'] for r in h1])
        d['DU_only_detections']=int(np.sum(~cc_pred&du_pred))
        d['classic_only_detections']=int(np.sum(cc_pred&~du_pred))
        summaries.append(d)
    for m in ('cc','du'):
        v=np.array([r[m+'_score'] for r in h0]);
        train,held=v[:1000],v[1000:]
        t=float(np.quantile(train,.99,method='higher'))
        cal.append({'method':m,'H0_train':len(train),'H0_temporal_holdout':len(held),
                    'threshold_train_0p01':t,'holdout_false_alarms':int(np.sum(held>t)),
                    'holdout_Pfa':float(np.mean(held>t)),
                    'all1500_in_sample_Pfa':float(np.mean(v>np.quantile(v,.99,method='higher')))})
    emit_csv('SUMMARY_1500H0_150H1.csv',summaries)
    emit_csv('TEMPORAL_H0_DIAGNOSTIC.csv',cal)
    rts=[]
    for m in ('cc','du'):
        vals=[r[m+'_ms'] for r in rows]
        rts.append({'method':m,'calls':len(vals),'median_ms':statistics.median(vals),
                    'p95_ms':float(np.quantile(vals,.95)), 'mean_ms':float(np.mean(vals))})
    emit_csv('RUNTIME_SUBSET.csv',rts)
    assert len(rows)==1800
    meta=parts[0]['meta'].copy();meta.pop('worker');meta['workers']=5
    (RS/'AUDIT.json').write_text(json.dumps({'status':'PASS', 'protocol':meta,'summary':summaries,
                           'temporal_H0_diagnostic':cal,'runtime':rts,'note':'Exploratory one-config subset; H0 1500 calibration, temporally dependent windows; no independent Pfa guarantee'},indent=2))
    print('PASS 1800 paired score rows, 1500 H0, 150 H1 per SCR, 5 worker shards')
    for d in summaries:
        print('SCR',d['SCR_dB'],'Pd:',d['cc_Pd_nominal_0p01'], d['du_Pd_nominal_0p01'], 'AUC:',d['cc_AUC_all_H0'],d['du_AUC_all_H0'],'discordant',d['DU_only_detections'],d['classic_only_detections'])
    print('H0 temporal:',cal)
    print('speedup medians',rts[0]['median_ms']/rts[1]['median_ms'])

if __name__=='__main__':main()
