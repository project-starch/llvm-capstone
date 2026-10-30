#!/usr/bin/env python3
"""objb-grade.py <run dir> <bs3|objb> [<other arm's run dir>]: grade one arm of the ObjBounds kit by value against
objb-bare-2026-10-30/PREREG.md + addendum 1 (llvm-capstone land/rtl-objb). Reads each image's uart.txt (SR/SV lines)
and run.log (image path, resident bitstream), hashes the image it loaded. With a second dir (the other arm), prints
the cost pattern board 2 against board 2. Exit 0 = AS EXPECTED / CONFIRMED; 1 = any other verdict; 2 = NO DATA."""
import sys, re, hashlib, pathlib
D = pathlib.Path(sys.argv[1]); ARM = sys.argv[2]; OTHER = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
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
want = BS3 if ARM == 'bs3' else OBJB; other = OBJB if ARM == 'bs3' else BS3
nodata = []; bad = []; stop = []; void = []
for n in H:
    g = read(D, n)
    if g is None:
        if n == 'objb-pcc' and ARM == 'bs3' and (D / 'objb-pcc').exists() is False:
            print(f'{n:14s} not run (bitstream 3: expected not to return)'); nodata.append(n); continue
        print(f'{n:14s} NO DATA'); nodata.append(n); continue
    hok = g['h'] == H[n]; e = want[n]
    if e is None:
        ok = hok and not g['sr']
        print(f'{n:14s} img {"OK" if hok else "MISMATCH"} bs {g["bs"]} SR {g["sr"]} -> {"no return, as expected" if ok else "RETURNED"}')
        if g['sr']: stop.append(n)
        continue
    exact = hok and g['sr'] == [len(e)] and g['sv'] == e
    ncy = CYC.get(n, [])
    noncyc_ok = hok and g['sr'] == [len(e)] and len(g['sv']) == len(e) and all(g['sv'][i] == e[i] for i in range(len(e)) if i not in ncy)
    print(f'{n:14s} img {"OK" if hok else "MISMATCH"} bs {g["bs"]} SR {g["sr"]} SV {" ".join(hex(x) for x in g["sv"])}'
          + ('  EXACT' if exact else ('  non-cycle OK' if noncyc_ok else '  OFF')))
    if n == 'objb-pcc-ctrl' and not exact: void.append(n)
    elif n == 'objb-stale' and not exact:
        (stop if g['sv'] == other[n] else bad).append(n)
    elif n == 'objb-pcc' and not exact: bad.append(n)
    elif n in CYC and not noncyc_ok: bad.append(n)
v = 'VOID' if void else ('STOP' if stop else ('FINDING' if bad else ('NO DATA' if [x for x in nodata if not (x == 'objb-pcc' and ARM == 'bs3')] else
    ('AS EXPECTED' + (' on 9 of 10 (objb-pcc not run, so the fetch face is unread)' if 'objb-pcc' in nodata else '') if ARM == 'bs3' else 'CONFIRMED'))))
print(f'ARM {ARM}: {v}  (void {void} stop {stop} finding {bad} no-data {nodata})')
if OTHER:
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
sys.exit(0 if v == 'AS EXPECTED' or v == 'CONFIRMED' else (2 if v == 'NO DATA' else 1))
