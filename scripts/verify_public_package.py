#!/usr/bin/env python3
"""Check integrity and study separation for clean public DU-CCEBM candidate (no raw required)."""
import hashlib, json, pathlib, sys
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_manifest(path,base):
    total=0
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        h,rel=line.split('  ',1)
        p=base/rel
        assert p.is_file(),f'Missing file: {p}'
        assert sha(p)==h,f'Hash mismatch: {p}'
        total+=1
    return total
base=json.loads((R/'SOURCE_MANIFEST.json').read_text())
assert len(base)==41, f'Expected 41 original two-layer SHA checks, got {len(base)}'
for rec in base:
    assert sha(R/rec['file'])==rec['sha256'], rec['file']
four=R/'extensions/netrad-ipix-4layer'
six=R/'extensions/ipix-six-layer-frozen'
checks=check_manifest(four/'SHA256SUMS',four)
checks6=check_manifest(six/'SHA256SUMS',six)
whole=check_manifest(R/'PUBLIC_SHA256SUMS',R)
assert sha(six/'checkpoints/real_A_best_model.pt')=='5d450c073c22c3cb9aeca8b49fe837549f3921d347439b3db268f3234981fb01'
forbidden={'.cdf','.mat'}
assert not any(p.suffix.lower() in forbidden for p in R.rglob('*') if p.is_file())
print(f'PASS original two-layer manifest: {len(base)}; four-layer: {checks}; six-layer: {checks6}; public package: {whole}')
print('PASS checkpoint digest and no raw IPIX/NetRAD CDF/MAT files')
