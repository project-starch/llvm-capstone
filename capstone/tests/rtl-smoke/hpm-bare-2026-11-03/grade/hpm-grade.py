#!/usr/bin/env python3
"""hpm-grade.py <run dir> <bs3|objb|c5a> [<other arm's run dir>]: grade one arm of the hpm kit against
hpm-bare-2026-11-03/PREREG.md. Reads <run dir>/<image>/uart.txt (SR/SV lines) and run.log (image path, resident
bitstream) for each of hpm-1..7, and hashes the image that run loaded.

Field order of every vector: load, store, l1d_acc, l1d_miss, branch, stall, cycles, instret, ev3, traps, cause.
- EXACT fields must equal the pre-registered vector: load, store, l1d_miss, branch, instret, ev3, traps, cause, and
  every field of hpm-6 (the inhibit arm) except its cycles (corrected after board 1's reading, see PREREG).
- TIMING fields are cycles, stall and l1d_acc (outside hpm-6). They are reported, not required, because silicon memory
  latency is not the simulator's delay 12.
With the other arm's directory it also grades the cost pattern:
- hpm-3 and hpm-4 have lower stall and cycles on ObjBounds, and the stall drop is within 15 % of the cycle drop;
- every other image's cycles are equal across the two bitstreams within 1 %.
The resident bitstream name read back in run.log must name the arm (e7af48457 / 603528b8b), else WRONG BITSTREAM.
ARM c5a (added 2026-11-04, before any C.5a boot): the C.5a bitstream (capstone-ariane 1fc2cd5aa). Its vectors were
simulated from these images on c5a-wt a642f4891 (core = 1fc2cd5aa, rtl_dirty=no); its exact fields equal ObjBounds'.
Its cost pattern is graded against a bitstream 3 run, like ObjBounds'. Resident name must contain 1fc2cd5aa.
Exit 0 = AS EXPECTED; 1 = any other verdict; 2 = NO DATA."""
import sys, re, hashlib, pathlib
D = pathlib.Path(sys.argv[1]); ARM = sys.argv[2]; OTHER = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
if ARM == 'c5a' and OTHER is None:
    # Every exact field reads the same on bitstream 3 and C.5a, so only the cost pattern against a bitstream 3 run
    # separates them: without that run the c5a arm has nothing to grade.
    print('NO DATA: arm c5a needs the bitstream 3 run dir as the third argument'); sys.exit(2)
if ARM not in ('bs3', 'objb', 'c5a'): sys.exit('arm must be bs3, objb or c5a')
H = {'hpm-5': 'dcc339ee4c1940ff0ef58ef0b98ccc5be84011f01e232113f59e1c790b61a004',
     'hpm-1': '4a16a2203c8ba9c4547f2b1b6d66ad805cdef915c62fbc9fe4936bfc868d0b34',
     'hpm-2': '22b8c6bf2ae23af45ad8dfef77d11103790c394301fdbb5db15dec9fe2fc0ecc',
     'hpm-3': '87938ddad16920fcb1e42f66ba366b2397c976f5ea5a7f38baf997b2e98c588a',
     'hpm-4': '1bcb3379d9c00ab087bc24a676ce352af49844b9fce163f67295f4502c9a7cbb',
     'hpm-6': '0bf4d19bbe4cd1304dcad5376f88ebd9488c4f037d552ef697c3702fa98356de',
     'hpm-7': '29d77f372692a5cd1b7245e951649d280b97ee425bf9f487ae02ff08a5388029'}
V = lambda s: [int(x, 16) for x in s.split()]
BS3 = {'hpm-5': V('0x0 0x0 0x0 0x0 0x100 0x0 0x33a 0x203 0x5 0x0 0x0'),
       'hpm-1': V('0x100 0x0 0x100 0x0 0x100 0x0 0x468 0x303 0x5 0x0 0x0'),
       'hpm-2': V('0x0 0x100 0x100 0x0 0x100 0x0 0x504 0x303 0x5 0x0 0x0'),
       'hpm-3': V('0x0 0x0 0x100 0x0 0x100 0x5fa 0x95a 0x303 0x5 0x0 0x0'),
       'hpm-4': V('0x0 0x0 0x100 0x0 0x100 0x5fa 0xc10 0x303 0x5 0x0 0x0'),
       'hpm-6': V('0x0 0x0 0x0 0x0 0x0 0x0 0x493 0x303 0x5 0x0 0x0'),
       'hpm-7': V('0x100 0x0 0x100 0x100 0x100 0x0 0x5155 0x403 0x5 0x0 0x0')}
OBJB = dict(BS3)
OBJB['hpm-3'] = V('0x0 0x0 0x100 0x0 0x100 0x1ff 0x565 0x303 0x5 0x0 0x0')
OBJB['hpm-4'] = V('0x0 0x0 0x100 0x0 0x100 0x100 0x71c 0x303 0x5 0x0 0x0')
C5A = {'hpm-5': V('0x0 0x0 0x0 0x0 0x100 0x0 0x333 0x203 0x5 0x0 0x0'),
       'hpm-1': V('0x100 0x0 0x100 0x0 0x100 0x0 0x461 0x303 0x5 0x0 0x0'),
       'hpm-2': V('0x0 0x100 0x100 0x0 0x100 0x0 0x4fd 0x303 0x5 0x0 0x0'),
       'hpm-3': V('0x0 0x0 0x100 0x0 0x100 0x1ff 0x55e 0x303 0x5 0x0 0x0'),
       'hpm-4': V('0x0 0x0 0x100 0x0 0x100 0x100 0x715 0x303 0x5 0x0 0x0'),
       'hpm-6': V('0x0 0x0 0x0 0x0 0x0 0x0 0x48c 0x303 0x5 0x0 0x0'),
       'hpm-7': V('0x100 0x0 0x100 0x100 0x100 0x0 0x514e 0x403 0x5 0x0 0x0')}
TIMING = {2, 5, 6}
STALL, CYC = 5, 6
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
want = {'bs3': BS3, 'objb': OBJB, 'c5a': C5A}[ARM]
nodata, bad, wrongbs = [], [], []
RES = {'bs3': 'e7af48457', 'objb': '603528b8b', 'c5a': '1fc2cd5aa'}
for n in H:
    g = read(D, n)
    if g is None: print(f'{n}: NO DATA'); nodata.append(n); continue
    e = want[n]; hok = g['h'] == H[n]
    # CORRECTED 2026-11-03 after board 1's reading: hpm-6's CYCLES is a timing field like every other image's; the
    # inhibit arm's claim is its six counters (and l1d_acc, stall) reading 0, which stay exact. Was: every field exact.
    exact_idx = [i for i in range(len(e)) if i != CYC] if n == 'hpm-6' else [i for i in range(len(e)) if i not in TIMING]
    ok = hok and g['sr'] == [len(e)] and len(g['sv']) == len(e) and all(g['sv'][i] == e[i] for i in exact_idx)
    tim = '' if not g['sv'] or len(g['sv']) != len(e) else '  timing ' + ' '.join(
        f'{["", "", "l1d_acc", "", "", "stall", "cycles"][i]} {g["sv"][i]} (sim {e[i]})' for i in sorted(TIMING))
    print(f'{n}: img {"OK" if hok else "MISMATCH"} bs {g["bs"]} SR {g["sr"]} SV {" ".join(hex(x) for x in g["sv"])}'
          f'  {"EXACT FIELDS OK" if ok else "OFF"}{tim}')
    if not ok: bad.append(n)
    if not g['bs'] or RES[ARM] not in g['bs'][-1]: wrongbs.append(n)
v = 'NO DATA' if nodata else ('WRONG BITSTREAM' if wrongbs else ('FINDING' if bad else 'AS EXPECTED'))
print(f'ARM {ARM}: {v}  (off {bad} wrong-bitstream {wrongbs} no-data {nodata})')
patt_ok = True
# comparison-dir bitstream check (2026-11-04): the other run dir must have run on the bitstream its pattern is graded
# against, read back from its own run.log; without this a comparison against the wrong bitstream grades as a pattern.
CMP = {'bs3': ('603528b8b',), 'objb': ('e7af48457',), 'c5a': ('e7af48457',)}
if OTHER:
    seen = sorted({g['bs'][-1] for g in (read(OTHER, n) for n in H) if g and g['bs']})
    cmp_ok = bool(seen) and all(any(h in s for h in CMP[ARM]) for s in seen)
    print(f'COMPARISON DIR bitstream {seen}: ' + ('OK' if cmp_ok else 'WRONG COMPARISON BITSTREAM (must name ' + ' or '.join(CMP[ARM]) + ')'))
    patt_ok = cmp_ok
if OTHER and v == 'AS EXPECTED':
    a3, ao = (D, OTHER) if ARM == 'bs3' else (OTHER, D)
    print(f'COST PATTERN (bitstream 3 -> {"C.5a" if ARM == "c5a" else "ObjBounds"}):')
    for n in H:
        x, y = read(a3, n), read(ao, n)
        if not x or not y or len(x['sv']) != 11 or len(y['sv']) != 11: print(f'  {n}: NO DATA'); patt_ok = False; continue
        if n in ('hpm-3', 'hpm-4'):
            ds, dc = x['sv'][STALL] - y['sv'][STALL], x['sv'][CYC] - y['sv'][CYC]
            ok = ds > 0 and dc > 0 and abs(ds - dc) <= 0.15 * dc
            print(f'  {n}: stall {x["sv"][STALL]} -> {y["sv"][STALL]} (-{ds}), cycles {x["sv"][CYC]} -> {y["sv"][CYC]} (-{dc}),'
                  f' per op {ds / 256:.2f} / {dc / 256:.2f}: {"HOLDS" if ok else "FAILS"}')
        else:
            r = y['sv'][CYC] / x['sv'][CYC]; ok = abs(r - 1) <= 0.01
            print(f'  {n}: cycles {x["sv"][CYC]} -> {y["sv"][CYC]} ratio {r:.4f}, predicted equal within 1 %: {"HOLDS" if ok else "FAILS"}')
        patt_ok &= ok
    print(f'COST PATTERN: {"HOLDS" if patt_ok else "FAILS"}')
sys.exit(2 if v == 'NO DATA' else (0 if v == 'AS EXPECTED' and patt_ok else 1))
