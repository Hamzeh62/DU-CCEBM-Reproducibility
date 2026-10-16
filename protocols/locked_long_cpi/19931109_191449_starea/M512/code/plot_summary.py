from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
with (ROOT/'results/SUMMARY_1500H0_150H1.csv').open() as f:r=list(csv.DictReader(f))
xx=list(range(len(r))); w=.35
fig,ax=plt.subplots(figsize=(7,4.4))
ax.bar([i-w/2 for i in xx],[float(z['cc_Pd_nominal_0p01']) for z in r],width=w,label='CCEBM-Full-PU')
ax.bar([i+w/2 for i in xx],[float(z['du_Pd_nominal_0p01']) for z in r],width=w,label='DU-CCEBM (2 layers)')
ax.set_xticks(xx,[str(z['SCR_dB'])+' dB' for z in r]);ax.set_ylabel('Detection probability')
ax.set_xlabel('SCR');ax.set_ylim(0,1.02)
ax.set_title('IPIX, M=512, Doppler offset 60 Hz\nNominal Pfa=0.01; H0 calibration 1500, H1 150/SCR')
ax.legend(loc='best');fig.tight_layout()
fig.savefig(ROOT/'figures/PD_M512_H0_1500.png',dpi=180)
print('FIGURE_SAVED',ROOT/'figures/PD_M512_H0_1500.png')
