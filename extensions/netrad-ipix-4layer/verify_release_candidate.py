from __future__ import annotations
import csv,hashlib,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parent
ARMS={"du_ccebm","ccebm_pu","ccebm_wl","ccebm_dp","stfica","kelly_glrt","amf_glrt","anmf","ca_cfar"}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def err(s): raise AssertionError(s)
golden=json.loads((R/'docs/NUMERIC_GOLDEN_SHA256.json').read_text())
for rec in golden:
 path=R/rec['path']
 assert path.is_file() and sha(path)==rec['sha256'], ('numeric CSV was modified',path)
checks=sorted((R/'NetRAD/data/checkpoints_du').glob('netrad_roc_*_v1.json'))
assert len(checks)==21
for p in checks:
 ob=json.loads(p.read_text())
 assert ob['complete'] is True
 assert set(ob['h0'])==ARMS, (p,'h0',set(ob['h0']))
 assert set(ob['thr']).issubset(ARMS), (p,'thr',set(ob['thr']))
 assert len(ob['h0']['du_ccebm'])==400
 for v in ob['levels'].values():
  assert set(v)==ARMS and len(v['du_ccebm'])==80
ip=json.loads((R/'IPIX/data/ipix_sweep_v4duc.json').read_text())
assert len(ip['files'])==8
for k,f in ip['files'].items():
 for lv in f['levels'].values():
  assert not any('cerbm' in a and 'ccebm' not in a for a in lv),k
  for a in ARMS: assert a+'_M128' in lv,(k,a)
 for v in f['thr'].values():
  assert not any('cerbm' in a and 'ccebm' not in a for a in v),k
assert 'cerbm' not in ip['provenance']['blind_arms']
print('PASS: 21 NetRAD score checkpoints; 8 IPIX acquisition summaries; 9 retained arms; 8 unchanged numeric CSV SHA-256 digests; no CERBM-family numeric results')
