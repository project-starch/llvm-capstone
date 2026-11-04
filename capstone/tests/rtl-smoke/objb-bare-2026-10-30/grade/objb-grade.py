#!/usr/bin/env python3
"""objb-grade.py <run dir> <bs3|objb|c5a> [<other arm's run dir>]: grade one arm of the ObjBounds kit by value against
objb-bare-2026-10-30/PREREG.md + addendum 1 (llvm-capstone land/rtl-objb). Reads each image's uart.txt (SR/SV lines)
and run.log (image path, resident bitstream), hashes the image it loaded. With a second dir (the other arm), prints
the cost pattern board 2 against board 2. Exit 0 = AS EXPECTED / CONFIRMED; 1 = any other verdict; 2 = NO DATA.
ARM c5a (added 2026-11-04, before any C.5a boot): the C.5a bitstream (capstone-ariane 1fc2cd5aa), vectors simulated
from these images on c5a-wt a642f4891 (core = 1fc2cd5aa, rtl_dirty=no). The identity images read bitstream 3's
vectors, and objb-pcc does not return, because C.5a keeps every lifetime check. The witness objb-stale-w is graded
too, against bitstream 3's 15 values. The cost images' non-cycle positions are exact. With a bitstream 3 run as the
other dir, the cycle positions are graded by direction (C5A_DIR). The resident name must contain 1fc2cd5aa, else
WRONG BITSTREAM."""
import sys, re, hashlib, pathlib
D = pathlib.Path(sys.argv[1]); ARM = sys.argv[2]; OTHER = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
if ARM == 'c5a' and OTHER is None:
    # Every exact field reads the same on bitstream 3 and C.5a, so only the cost pattern against a bitstream 3 run
    # separates them: without that run the c5a arm has nothing to grade.
    print('NO DATA: arm c5a needs the bitstream 3 run dir as the third argument'); sys.exit(2)
H = {'objb-pcc-ctrl': 'fc23a616026e6d10830bc8cb38be52763ef92456bb93859203cee11bceffba96',
     'objb-stale': '08b1d9aabb41f6893def5940a23a89baab0a7cf4ee295b259e496c67d1a323df',
     'objb-perf-s1': '8073369ae0aa70b7926c423405bc4555775eb3adb3a2e2574ba0543f6269c4f3',
     'objb-perf-s2': '2f22d39e3069f915b21d29e2319e565248dc955b3ff1a96c2b3f4799d84bb36a',
     'objb-perf-s3': 'dc331d90d518d0e65f84a45bff975a4823c4e8967092eaaa4f94f36e5cb4cd79',
     'objb-perf-s4': '71a2464eb09747370af2cb77466003bc8b9759dac74ec332070b42889d244288',
     'objb-perf-s5': '9ac3442093cc21363d2e71cc1a401466be613b137c9e4ff8b84bcf8f76cd0506',
     'objb-perf-s6': 'ebc207552864ae09d5112540096e31eb26a107dd19d26928f77fb0c376dd0a3a',
     'objb-perf-s7': '4acb86e5e22cec12babe0721d7fa27af225fcdae2977fd644bb7cfa5b0b716f5',
     'objb-pcc': '0863f5636359a978a94b45fb7c3debf51955b358aa413060d24f8580d1251ac6'}
V = lambda s: [int(x, 16) for x in s.split()]
if ARM == 'c5a':
    H['objb-stale-w'] = 'ae0e1cb35dcc42d3c7bf5e370c6b1ec68a1ed1b9d2305b0a9e9f94191357e582'
BS3 = {'objb-pcc-ctrl': V('0xbb5 0x0 0x0'), 'objb-stale': V('0x19 0x19 0x19 0x19 0x0 0x19 0x19 0x77 0x6'),
       'objb-perf-s1': V('0x414f 0x1403 0x5a 0x0 0x0'), 'objb-perf-s2': V('0x2423 0xc03 0x80004000 0x0 0x0'),
       'objb-perf-s3': V('0x2b709 0x1804 0x80004000 0x0 0x0'), 'objb-perf-s4': V('0x290 0xc3 0x2c4 0xc3 0x80004000 0x0 0x0'),
       'objb-perf-s5': V('0xa0d8f 0x4da4d 0x400a 0xffc00 0x0 0x0'), 'objb-perf-s6': V('0x2430 0x503 0x0 0x0 0x0'),
       'objb-perf-s7': V('0x2c72 0x503 0x80004000 0x0 0x0'), 'objb-pcc': None}
OBJB = {'objb-pcc-ctrl': V('0xbb5 0x0 0x0'), 'objb-stale': V('0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x3333 0x0'),
        'objb-perf-s1': V('0x414f 0x1403 0x5a 0x0 0x0'), 'objb-perf-s2': V('0x145f 0xc03 0x80004000 0x0 0x0'),
        'objb-perf-s3': V('0x22744 0x1804 0x80004000 0x0 0x0'), 'objb-perf-s4': V('0x1d0 0xc3 0x1d0 0xc3 0x80004000 0x0 0x0'),
        'objb-perf-s5': V('0xa0c5f 0x3b049 0x400a 0xffc00 0x0 0x0'), 'objb-perf-s6': V('0x1422 0x503 0x0 0x0 0x0'),
        'objb-perf-s7': V('0x18c4 0x503 0x80004000 0x0 0x0'), 'objb-pcc': V('0xbb5 0x0 0x0')}
OBJB['objb-stale-w'] = V('0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x3333 0x0 0x0 0x19 0x19 0x19 0x2222 0x3')  # addendum 2
C5A = {'objb-pcc-ctrl': V('0xbb5 0x0 0x0'), 'objb-stale': V('0x19 0x19 0x19 0x19 0x0 0x19 0x19 0x77 0x6'),
       'objb-stale-w': V('0x19 0x19 0x19 0x19 0x0 0x19 0x19 0x77 0x6 0x0 0x19 0x19 0x19 0x0 0x9'),
       'objb-perf-s1': V('0x4156 0x1403 0x5a 0x0 0x0'), 'objb-perf-s2': V('0x145f 0xc03 0x80004000 0x0 0x0'),
       'objb-perf-s3': V('0x2c76a 0x1804 0x80004000 0x0 0x0'), 'objb-perf-s4': V('0x5b2 0xc3 0x1d0 0xc3 0x80004000 0x0 0x0'),
       'objb-perf-s5': V('0xa0c5f 0x4ca48 0x400a 0xffc00 0x0 0x0'), 'objb-perf-s6': V('0x1429 0x503 0x0 0x0 0x0'),
       'objb-perf-s7': V('0x18ca 0x503 0x80004000 0x0 0x0'), 'objb-pcc': None}
# C.5a against bitstream 3, per cycle position: 'eq' = equal within 1 %, 'lo' = lower on C.5a, 'hi' = higher on C.5a,
# 'rep' = reported only (s5 pass 2: 1.3 % lower in simulation, too close to call on silicon).
C5A_DIR = {('objb-perf-s1', 0): 'eq', ('objb-perf-s2', 0): 'lo', ('objb-perf-s3', 0): 'hi', ('objb-perf-s4', 0): 'hi',
           ('objb-perf-s4', 2): 'lo', ('objb-perf-s5', 0): 'eq', ('objb-perf-s5', 1): 'rep', ('objb-perf-s6', 0): 'lo',
           ('objb-perf-s7', 0): 'lo'}
CYC = {'objb-perf-s1': [0], 'objb-perf-s2': [0], 'objb-perf-s3': [0], 'objb-perf-s4': [0, 2], 'objb-perf-s5': [0, 1],
       'objb-perf-s6': [0], 'objb-perf-s7': [0]}
def read(d, n):
    u = d / n / 'uart.txt'; r = d / n / 'run.log'
    if not u.exists() or not r.exists(): return None
    t = u.read_text(errors='replace'); lg = r.read_text(errors='replace')
    sr = [int(x, 16) for x in re.findall(r'^SR ([0-9A-Fa-f]{16})\s*$', t, re.M)]
    sv = [int(x, 16) for x in re.findall(r'^SV ([0-9A-Fa-f]{16})\s*$', t, re.M)]
    m = re.search(r'image (\S+\.bin) ->', lg); img = pathlib.Path(m.group(1)) if m else None
    h = hashlib.sha256(img.read_bytes()).hexdigest() if img and img.exists() else None
    bs = re.findall(r"resident NV bitstream = '([^']+)'", lg)
    return dict(sr=sr, sv=sv, h=h, bs=bs)
want = {'bs3': BS3, 'objb': OBJB, 'c5a': C5A}[ARM]; other = OBJB if ARM in ('bs3', 'c5a') else BS3
wrongbs = []
nodata = []; bad = []; stop = []; void = []
for n in H:
    g = read(D, n)
    if g is None:
        if n == 'objb-pcc' and ARM in ('bs3', 'c5a') and (D / 'objb-pcc').exists() is False:
            print(f'{n:14s} not run (bitstream 3: expected not to return)'); nodata.append(n); continue
        print(f'{n:14s} NO DATA'); nodata.append(n); continue
    hok = g['h'] == H[n]; e = want[n]
    if ARM == 'c5a' and (not g['bs'] or '1fc2cd5aa' not in g['bs'][-1]): wrongbs.append(n)
    if e is None:
        ok = hok and not g['sr']
        print(f'{n:14s} img {"OK" if hok else "MISMATCH"} bs {g["bs"]} SR {g["sr"]} -> {"no return, as expected" if ok else "RETURNED"}')
        if g['sr']: (bad if ARM == 'c5a' else stop).append(n)
        continue
    exact = hok and g['sr'] == [len(e)] and g['sv'] == e
    ncy = CYC.get(n, [])
    noncyc_ok = hok and g['sr'] == [len(e)] and len(g['sv']) == len(e) and all(g['sv'][i] == e[i] for i in range(len(e)) if i not in ncy)
    print(f'{n:14s} img {"OK" if hok else "MISMATCH"} bs {g["bs"]} SR {g["sr"]} SV {" ".join(hex(x) for x in g["sv"])}'
          + ('  EXACT' if exact else ('  non-cycle OK' if noncyc_ok else '  OFF')))
    if n == 'objb-pcc-ctrl' and not exact: void.append(n)
    elif n in ('objb-stale', 'objb-stale-w') and not exact:
        (stop if g['sv'] == other[n] else bad).append(n)
    elif n == 'objb-pcc' and not exact: bad.append(n)
    elif n in CYC and not noncyc_ok: bad.append(n)
v = 'WRONG BITSTREAM' if wrongbs else 'VOID' if void else ('STOP' if stop else ('FINDING' if bad else ('NO DATA' if [x for x in nodata if not (x == 'objb-pcc' and ARM in ('bs3', 'c5a'))] else
    ('AS EXPECTED' + (' on 9 of 10 (objb-pcc not run, so the fetch face is unread)' if 'objb-pcc' in nodata else '') if ARM in ('bs3', 'c5a') else 'CONFIRMED'))))
print(f'ARM {ARM}: {v}  (wrong-bitstream {wrongbs} void {void} stop {stop} finding {bad} no-data {nodata})')
patt_ok = True
# comparison-dir bitstream check (2026-11-04): the other run dir must have run on the bitstream its pattern is graded
# against, read back from its own run.log; without this a comparison against the wrong bitstream grades as a pattern.
CMP = {'bs3': ('603528b8b',), 'objb': ('e7af48457',), 'c5a': ('e7af48457',)}
if OTHER:
    seen = sorted({g['bs'][-1] for g in (read(OTHER, n) for n in H) if g and g['bs']})
    cmp_ok = bool(seen) and all(any(h in s for h in CMP[ARM]) for s in seen)
    print(f'COMPARISON DIR bitstream {seen}: ' + ('OK' if cmp_ok else 'WRONG COMPARISON BITSTREAM (must name ' + ' or '.join(CMP[ARM]) + ')'))
    patt_ok = cmp_ok
if OTHER and ARM == 'c5a':
    print('COST PATTERN, C.5a against bitstream 3, cycles:')
    for (n, i), d in C5A_DIR.items():
        x, y = read(OTHER, n), read(D, n)
        if not x or not y or len(x['sv']) <= i or len(y['sv']) <= i: print(f'  {n}[{i}]: NO DATA'); patt_ok = False; continue
        r = y['sv'][i] / x['sv'][i]
        ok = {'eq': abs(r - 1) <= 0.01, 'lo': r < 1, 'hi': r > 1, 'rep': True}[d]
        print(f'  {n}[{i}]: {x["sv"][i]} -> {y["sv"][i]} ratio {r:.4f}  predicted {d}: {"HOLDS" if ok else "FAILS"}')
        patt_ok &= ok
    print(f'COST PATTERN: {"HOLDS" if patt_ok else "FAILS"}')
elif OTHER:
    print('COST PATTERN, board 2 vs board 2 (bitstream 3 -> ObjBounds), cycles:')
    a3 = D if ARM == 'bs3' else OTHER; ao = OTHER if ARM == 'bs3' else D
    for n, cy in CYC.items():
        x, y = read(a3, n), read(ao, n)
        if not x or not y or not x['sv'] or not y['sv']: print(f'  {n}: NO DATA'); continue
        for i in cy:
            r = y['sv'][i] / x['sv'][i]
            eq = (n == 'objb-perf-s1' and i == 0) or (n == 'objb-perf-s5' and i == 0)
            pred = 'equal within 1%' if eq else 'lower on ObjBounds'
            ok = abs(r - 1) <= 0.01 if eq else r < 1
            print(f'  {n}[{i}]: {x["sv"][i]} -> {y["sv"][i]} ratio {r:.4f}  predicted {pred}: {"HOLDS" if ok else "FAILS"}')
sys.exit(2 if v == 'NO DATA' else (0 if (v == 'AS EXPECTED' or v == 'CONFIRMED') and patt_ok else 1))
