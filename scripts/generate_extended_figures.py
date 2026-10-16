#!/usr/bin/env python3
"""Curated figures from stored REAL IPIX results only; no new radar scoring."""
import pathlib, csv, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'figures';OUT.mkdir(parents=True,exist_ok=True)
with (ROOT/'evidence/historical/longcpi_8SCR_21_60_exploratory_aggregate.csv').open(newline='') as f: rows=list(csv.DictReader(f))
fig, axes=plt.subplots(2,2, figsize=(11,7.2), sharey=True, sharex=True)
for ax,(M,off) in zip(axes.ravel(),[(512,21),(512,60),(1024,21),(1024,60)]):
 for method in ('CCEBM-Full-PU','DU-CCEBM depth-2'):
  matches=sorted([r for r in rows if int(r['M'])==M and int(r['offset'])==off and r['method']==method],key=lambda r:int(r['SCR']))
  ax.plot([int(r['SCR']) for r in matches],[float(r['Pd_descriptive']) for r in matches],marker='o',label=method)
 ax.set_title(f'M={M}, Doppler offset={off} Hz');ax.grid(alpha=.25);ax.set_ylim(0,1.03)
 ax.set_xlabel('SCR (dB)');ax.set_ylabel('Detection probability (descriptive)')
axes[0,0].legend(fontsize=8)
fig.suptitle('IPIX exploratory 8-SCR power curves; nominal Pfa=0.05, not validated CFAR',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'IPIX_8SCR_21_60_descriptive.pdf',bbox_inches='tight');plt.close(fig)

fig, axes=plt.subplots(2,2,figsize=(11,7.2),sharex=True,sharey=True)
for ax,(M,scr) in zip(axes.ravel(),[(512,-25),(512,-20),(1024,-25),(1024,-20)]):
 for method,field in [('CCEBM-Full-PU','cc_score'),('DU-CCEBM depth-2','du_score')]:
  curves=[]
  for recording in ('19931108_213827_starea','19931109_191449_starea'):
   p=ROOT/'evidence/long_cpi'/f'paired_scores_{recording}_M{M}.csv'
   with p.open(newline='') as f: sub=list(csv.DictReader(f))
   z=np.array([float(r[field]) for r in sub if r['kind']=='H0'])
   t=np.array([float(r[field]) for r in sub if r['kind']=='H1' and int(float(r['scr']))==scr])
   # Common nominal FPR operating-point grid; per-acquisition separate thresholding.
   grid=np.r_[np.array([0.001]),np.linspace(0.005,.10,50)]
   pd=np.array([np.mean(t>np.quantile(z,1-alpha,method='higher')) for alpha in grid])
   curves.append(pd)
  ax.plot(grid,np.mean(curves,axis=0),label=method)
 ax.plot([0,.1],[0,.1],linestyle=':',c='gray',linewidth=.7)
 ax.set_xscale('log');ax.set_xlim(.001,.10);ax.set_ylim(0,1.03)
 ax.set_title(f'M={M}, SCR={scr} dB');ax.grid(alpha=.25);ax.set_xlabel('Nominal false-alarm level');ax.set_ylabel('Detection probability')
axes[0,0].legend(fontsize=8)
fig.suptitle('IPIX locked long-CPI nominal-threshold curves; temporally dependent H0 windows',fontsize=10)
fig.tight_layout();fig.savefig(OUT/'IPIX_2Layer_Locked_Roc_nominal.pdf',bbox_inches='tight');plt.close(fig)
print('PASS 2 IPIX-only figures generated from packaged numeric evidence:',OUT)
