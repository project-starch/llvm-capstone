#!/usr/bin/env python3
"""judge-m2.py <qemu-log> <boot dir>...   M2's dependent chase against its pre-registration (m2-sublet-2026-10-16).

Reads each boot's r1-lines.txt (the r1e4 driver's marker-free, seam-joined R1 lines) and its `log` summary.
Checks, per point: chk_ok=1 and chk equal to the emulator's for (arm, seed, N); cycle_ok=1 on every cold traversal;
minted D/P 6, S 2N+4. Reports cycles per access (cyc/loads) per arm and N, the median over the runs, and S/P.
Exit: 0 every check holds; 1 a check failed (listed); 2 no data.
"""
import re, sys, os, statistics, collections

def kv(l): return {k: int(v) if re.fullmatch(r'-?\d+', v) else v for k, v in re.findall(r'(\w+)=(\S+)', l)}

if len(sys.argv) < 3: sys.exit(__doc__)
q = open(sys.argv[1], errors='replace').read()
oracle = {}
for l in q.splitlines():
    if l.startswith('R1 chase arm='):
        d = kv(l); oracle[(d['arm'], d['seed'], d['N'])] = d['chk']
def nodata(m): print('NO DATA: ' + m); sys.exit(2)
if len(oracle) != 45: nodata(f'{len(oracle)} emulator points in {sys.argv[1]}, expected 45')

pts, cold, fails, boots = [], [], [], []
for b in sys.argv[2:]:
    f = os.path.join(b, 'r1-lines.txt')
    if not os.path.exists(f): fails.append(f'{b}: no r1-lines.txt'); continue
    lines = open(f, errors='replace').read().splitlines()
    log = open(os.path.join(b, 'log'), errors='replace').read() if os.path.exists(os.path.join(b, 'log')) else ''
    # the driver's summary prints `RESULT k800 retval=...  : <count>x [<distinct values>]`
    km = re.search(r"RESULT k800 retval=\S+\s*: (\d+)x \[([^\]]*)\]", log)
    k800 = (int(km.group(1)), sorted(set(re.findall(r"retval=(-?\d+)", km.group(2))))) if km else (0, [])
    marker = open(os.path.join(b, 'marker')).read().strip() if os.path.exists(os.path.join(b, 'marker')) else 'none'
    ch = [kv(l) for l in lines if l.startswith('R1 chase arm=')]
    cc = [kv(l) for l in lines if l.startswith('R1 chase-cold')]
    ref = [l for l in lines if l.startswith('R1 refused')]
    ends = [l for l in lines if l.startswith('R1 end')]
    boots.append((os.path.basename(b), marker, len(ch), len(cc), len(ends), len(ref)))
    if marker != 'done': fails.append(f'{os.path.basename(b)}: marker {marker}')
    if k800 != (2, ['4']): fails.append(f'{os.path.basename(b)}: k800 {k800[0]}x {k800[1]}, expected 4 twice')
    if ref: fails.append(f'{os.path.basename(b)}: {len(ref)} refused lines: {ref[0]}')
    for d in ch: d['boot'] = os.path.basename(b); pts.append(d)
    for d in cc: d['boot'] = os.path.basename(b); cold.append(d)
if not pts: nodata('no R1 chase lines in ' + ' '.join(sys.argv[2:]))

for d in pts:
    key = (d['arm'], d['seed'], d['N'])
    want_m = 2 * d['N'] + 4 if d['arm'] == 'S' else 6
    if d.get('chk_ok') != 1: fails.append(f"{d['boot']} {key}: chk_ok={d.get('chk_ok')}")
    if oracle.get(key) != d.get('chk'): fails.append(f"{d['boot']} {key}: chk {d.get('chk')} != emulator {oracle.get(key)}")
    if d.get('minted') != want_m: fails.append(f"{d['boot']} {key}: minted {d.get('minted')} != {want_m}")
for d in cold:
    if d.get('cycle_ok') != 1: fails.append(f"{d['boot']} cold {d.get('arm')} N={d.get('N')}: cycle_ok={d.get('cycle_ok')}")

per = collections.defaultdict(list)
for d in pts: per[(d['arm'], d['N'])].append(d['cyc'] / d['loads'])
Ns = sorted({n for _, n in per})
print('boots (name, marker, chase points, cold lines, R1 end, refused):')
for x in boots: print('  ', *x)
print(f'points: {len(pts)} (expected 225 = 45 invocations x 5 N); cold traversals: {len(cold)}')
print('cycles per access, median over runs [min..max] (runs):')
for n in Ns:
    row = []
    for a in 'DPS':
        v = per.get((a, n), [])
        row.append(f"{a} {statistics.median(v):8.3f} [{min(v):.3f}..{max(v):.3f}] ({len(v)})" if v else f'{a} -')
    p, s = per.get(('P', n)), per.get(('S', n))
    sp = f"S/P {statistics.median(s) / statistics.median(p):.4f}" if p and s else 'S/P -'
    print(f'  N={n:5d}  ' + '   '.join(row) + '   ' + sp)
if len(pts) != 225: fails.append(f'{len(pts)} points, expected 225')
print('CHECKS:', 'all hold' if not fails else f'{len(fails)} failed')
for f in fails[:40]: print('  FAIL', f)
sys.exit(1 if fails else 0)
