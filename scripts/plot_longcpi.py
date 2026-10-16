#!/usr/bin/env python3
"""Make a manuscript-ready plot from locked derived data; no new results."""
import argparse,csv,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/longcpi_power.pdf');a=p.parse_args()
with (R/'evidence/long_cpi/TWO_ACQUISITION_AGGREGATE.csv').open(newline='') as f:rows=list(csv.DictReader(f))
fig,axes=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
for M,ax in zip((512,1024),axes):
 rr=sorted([r for r in rows if int(r['M'])==M],key=lambda x:float(x['SCR_dB']))
 x=[float(r['SCR_dB']) for r in rr]
 ax.plot(x,[float(r['cc_mean_Pd']) for r in rr], 'o-',label='CCEBM-Full-PU')
 ax.plot(x,[float(r['du_mean_Pd']) for r in rr], 's-',label='DU-CCEBM (2 layers)')
 ax.set_title(f'CPI M={M}');ax.set_xlabel('SCR (dB)');ax.grid(True,alpha=.3);ax.set_ylim(0,1)
axes[0].set_ylabel('Detection probability')
axes[0].legend(fontsize=7)
fig.suptitle('IPIX, nominal false-alarm level 0.01 (not verified CFAR)',fontsize=10)
fig.tight_layout()
out=R/a.out;out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,bbox_inches='tight')
print('Wrote',out)
