#!/usr/bin/env python3
# status.py: every Tier 3 bundle run on the fixed v6 revision, from the verdict files on disk, compared with the record.
import json, subprocess, sys
M = '/tmp/capstone/vp2-prep/tier3/manifest.json'
F = '/tmp/capstone/vp2-prep/tier3-fix'
P = '/tmp/capstone/vp2-prep/tier3-pilot'
runs = [  # (corpus, arm, verdicts file, platform)
 ('ffmpeg/carved-repros', 'virtual-nested-pools', P+'/ffmpeg-carved-fix/virtual-nested-pools/verdicts.jsonl', 'kit 89ee98be'),
 ('ffmpeg/carved-repros', 'virtual-malloc', F+'/b5-ffmpeg-carved-repros/virtual-malloc/verdicts.jsonl', 'kit 89ee98be'),
 ('ffmpeg/plain-heap-repros', 'virtual-malloc', F+'/b7-ffmpeg-plain-heap-repros/virtual-malloc/verdicts.jsonl', 'kit 89ee98be'),
 ('ffmpeg/plain-temporal-repros', 'virtual-malloc', F+'/b8-ffmpeg-plain-temporal-repros/virtual-malloc/verdicts.jsonl', 'kit 89ee98be'),
 ('wireshark/plain-heap-repros', 'virtual-malloc', F+'/b23-wireshark-plain-heap-repros/virtual-malloc/verdicts.jsonl', 'kit 89ee98be'),
 ('wireshark/plain-temporal-repros', 'virtual-malloc', F+'/b24-wireshark-plain-temporal-repros/virtual-malloc/verdicts.jsonl', 'kit 89ee98be'),
 ('memcached/allocator-repros', 'virtual-malloc', F+'/b12-memcached-allocator-repros-virtual-malloc/virtual-malloc/verdicts.jsonl', 'kit 89ee98be'),
 ('memcached/plain-heap-repros', 'virtual-malloc', F+'/b-memcached-plain-heap-repros/virtual-malloc/verdicts.jsonl', 'kit 89ee98be'),
 ('memcached/plain-temporal-repros', 'virtual-malloc', F+'/b-memcached-plain-temporal-repros/virtual-malloc/verdicts.jsonl', 'kit 89ee98be'),
 ('postgres/c-repros', 'virtual-malloc', F+'/pgc/out/verdicts.jsonl', 'kit 89ee98be'),
 ('postgres/mmgr-repros', 'virtual-malloc', F+'/mmgr/out-virtual-malloc/verdicts.jsonl', 'kit 89ee98be'),
 ('postgres/mmgr-repros', 'virtual-pg-pools', F+'/mmgr/out-virtual-pg-pools/verdicts.jsonl', 'kit 89ee98be'),
 ('wireshark/wmem-repros', 'virtual-malloc', F+'/wmem/out-virtual-malloc/verdicts.jsonl', 'kit 89ee98be (capstone_vm)'),
 ('ffmpeg/subobject-repros', 'virtual-malloc', F+'/run6/subobject/virtual-malloc/verdicts.jsonl', 'SDK arm 6 (-fno-builtin)'),
 ('ffmpeg/pool-repros', 'virtual-malloc', F+'/run6/pool-OFF/virtual-malloc/verdicts.jsonl', 'SDK arm 6 (-fno-builtin)'),
 ('ffmpeg/pool-repros', 'virtual-nested-pools', F+'/run6/pool-ON/virtual-nested-pools/verdicts.jsonl', 'SDK arm 6 (-fno-builtin)'),
 ('memcached/allocator-repros', 'virtual-nested-pools', F+'/disc1-b13-exact/virtual-nested-pools/verdicts.jsonl', 'kit + QEMU exact bounds'),
 ('wireshark/wmem-repros', 'virtual-nested-pools', F+'/wmem/out-virtual-nested-pools-exact/verdicts.jsonl', 'kit + QEMU exact bounds (capstone_vm)'),
 ('sqlite/engine-repros', 'virtual-malloc', F+'/sqlite/out-virtual-malloc/verdicts.jsonl', 'kit + b0d93849ceba build fix (capstone_vm)'),
 ('sqlite/engine-repros', 'virtual-nested-pools', F+'/sqlite/out-virtual-nested-pools-exact/verdicts.jsonl', 'kit + b0d93849ceba + QEMU exact bounds (capstone_vm)'),
]
man = json.load(open(M)); allb = {(b['corpus'], b['arm']) for b in man['bundles']}
seen = set(); rows = []
for corpus, arm, vj, plat in runs:
    out = subprocess.run([sys.executable, '/tmp/capstone/vp2-prep/tier3/compare-cells.py', M, corpus, arm, vj], capture_output=True, text=True).stdout.splitlines()[0]
    rows.append((corpus, arm, plat, out.split(': ', 1)[1])); seen.add((corpus, arm))
for r in rows: print(' | '.join(r))
print(f"bundles run: {len(seen)} of {len(allb)}; not run: {sorted(allb - seen)}")
