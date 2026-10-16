#!/usr/bin/env python3
"""Verify exact extracted original code/checkpoint/data hashes (no raw CDF needed)."""
import json,hashlib,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1]
records=json.loads((R/'SOURCE_MANIFEST.json').read_text())
fail=[]
for item in records:
 p=R/item['file']
 if not p.is_file(): fail.append('missing: '+item['file']); continue
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h != item['sha256']:fail.append('modified: '+item['file'])
if fail:
 print('FAIL:',*fail,sep='\n');sys.exit(1)
assert all('netrad' not in item['file'].lower() for item in records)
print('PASS',len(records),'source-provenance checksums for frozen two-layer files (NetRAD extension is separate)')
