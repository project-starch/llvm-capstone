#!/usr/bin/env python3
"""slot-grade.py <run dir> <bs3|objb|c5a>: grade one run of the slot-bare kit against slot-bare-2026-11-04/PREREG.md.
Reads <run dir>/<image>/uart.txt (SR/SV lines) and run.log (image path, resident bitstream) for slot-1..9, and hashes
the image that run loaded.

Field order of every vector: cycles, instret, value, traps, cause.
- EXACT fields must equal the pre-registered vector: instret, value, traps, cause (the same on every bitstream).
- cycles is a TIMING field: reported beside the simulator's (memory delay 12), not required to match it. Each slot is
  also printed IN BAND / OUT OF BAND against its sim cycles +-10 % (CYCLE BAND, added 2026-11-04 from bitstream 3's
  silicon run, where every slot read 0.4-9 % under sim), with a summary line; it never changes the verdict, but a
  bitstream whose silicon does not track its own sim cannot pass silently.
- The silicon PATTERN, pre-registered from the source and the simulation of these images (bs3, objb, c5a):
  P1  slot-1, -2, -3, -4 (the stc/ldc pair: same slot, other doubleword, other line, page alias) read the same cycles
      within 2 %: the DYN unit's one-op handshake orders the pair, so the ldc never meets the buffered stc;
  P2  slot-5 (sd A; ld A) reads more cycles than slot-6 (sd A; ld A+0x10): the plain pair waits on the page offset.
- The dyntrim arm (stage A, capstone-ariane c9c7952bd, added 2026-11-04 before its first boot) replaces P1 with P1d:
  slot-1 and slot-4 (same doubleword, page alias) within 2 % of each other, slot-2 and slot-3 (other doubleword, other
  line) within 2 % of each other, and slot-2 more than 5 % below slot-1: with the handshake trimmed, a pair to another
  doubleword no longer waits, while the same doubleword (and its page alias) waits on the store's page offset.
- The per-op costs over the bare loop (slot-9) are printed: ldc, stc, the pair, the plain pairs.
The resident bitstream name read back in run.log must name the arm (e7af48457 / 603528b8b / 1fc2cd5aa), else WRONG
BITSTREAM (dyntrim: c9c7952bd). Exit 0 = AS EXPECTED (exact fields and both pattern clauses hold); 1 = any other verdict; 2 = NO DATA."""
import sys, re, hashlib, pathlib
D = pathlib.Path(sys.argv[1]); ARM = sys.argv[2] if len(sys.argv) > 2 else ''
RES = {'bs3': 'e7af48457', 'objb': '603528b8b', 'c5a': '1fc2cd5aa', 'dyntrim': 'c9c7952bd'}
if ARM not in RES: sys.exit('arm must be bs3, objb, c5a or dyntrim')
H = {'slot-9': 'ea3b5b4bd420d3425d408ca2bfb893e005799bbd610f29f32f58dc97a84dac3a',
     'slot-6': 'a451db608edb65803ad22a5867e51e97d4fa4d16129df84db0db3c4041651e3b',
     'slot-5': 'f4b2fa0c6ce869a37b1efa0398cb5526633fa864b05ce571ecbbbfd7ccbd2aa7',
     'slot-7': 'cda423944411c957fc48eac968c4088570274b41fcbd7f3eb61888f6cf503f13',
     'slot-8': 'bbbb2213f335203a0e1393625d263f518a4bebadeb0610710529374cac52c7af',
     'slot-1': '51e5573967ccd8be332b33085e362308e62b9ce4050a86418417dbfd8f1f4d31',
     'slot-2': '7ce9fb0e935c2e49e0d929f9049c6c8b1ff35c24ffa10fef3cdc23178212e6e2',
     'slot-3': 'ea3af885dfc49fb0aedb19c9325e3c67ee77d77b9fbaf5051bfd296e1d4e6ada',
     'slot-4': '469bb446f13e895630ce8655bff07c3b17cc0796c20b8ff15a5051547c459c1c'}
V = lambda s: [int(x, 16) for x in s.split()]
SIM = {'bs3': {'slot-9': V('0x639 0x403 0x0 0x0 0x0'), 'slot-6': V('0xb02 0x803 0x5a 0x0 0x0'), 'slot-5': V('0x12f7 0x803 0x1 0x0 0x0'), 'slot-7': V('0x1259 0x603 0x77 0x0 0x0'), 'slot-8': V('0x170f 0x603 0x0 0x0 0x0'), 'slot-1': V('0x28fd 0x803 0x77 0x0 0x0'), 'slot-2': V('0x28fd 0x803 0x77 0x0 0x0'), 'slot-3': V('0x28fd 0x803 0x77 0x0 0x0'), 'slot-4': V('0x28fd 0x803 0x77 0x0 0x0')},
       'objb': {'slot-9': V('0x639 0x403 0x0 0x0 0x0'), 'slot-6': V('0xb02 0x803 0x5a 0x0 0x0'), 'slot-5': V('0x12f7 0x803 0x1 0x0 0x0'), 'slot-7': V('0xa64 0x603 0x77 0x0 0x0'), 'slot-8': V('0xd1b 0x603 0x0 0x0 0x0'), 'slot-1': V('0x1712 0x803 0x77 0x0 0x0'), 'slot-2': V('0x1712 0x803 0x77 0x0 0x0'), 'slot-3': V('0x1712 0x803 0x77 0x0 0x0'), 'slot-4': V('0x1712 0x803 0x77 0x0 0x0')},
       'c5a': {'slot-9': V('0x632 0x403 0x0 0x0 0x0'), 'slot-6': V('0xafb 0x803 0x5a 0x0 0x0'), 'slot-5': V('0x12f0 0x803 0x1 0x0 0x0'), 'slot-7': V('0xa5d 0x603 0x77 0x0 0x0'), 'slot-8': V('0xd14 0x603 0x0 0x0 0x0'), 'slot-1': V('0x170b 0x803 0x77 0x0 0x0'), 'slot-2': V('0x170b 0x803 0x77 0x0 0x0'), 'slot-3': V('0x170b 0x803 0x77 0x0 0x0'), 'slot-4': V('0x170b 0x803 0x77 0x0 0x0')},
       'dyntrim': {'slot-9': V('0x632 0x403 0x0 0x0 0x0'), 'slot-6': V('0xafb 0x803 0x5a 0x0 0x0'), 'slot-5': V('0x12f0 0x803 0x1 0x0 0x0'), 'slot-7': V('0x860 0x603 0x77 0x0 0x0'), 'slot-8': V('0xd14 0x603 0x0 0x0 0x0'), 'slot-1': V('0x170a 0x803 0x77 0x0 0x0'), 'slot-2': V('0x1310 0x803 0x77 0x0 0x0'), 'slot-3': V('0x1310 0x803 0x77 0x0 0x0'), 'slot-4': V('0x170a 0x803 0x77 0x0 0x0')}}
N = 512
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
want = SIM[ARM]
nodata, bad, wrongbs, cyc = [], [], [], {}
for n in H:
    g = read(D, n)
    if g is None: print(f'{n}: NO DATA'); nodata.append(n); continue
    e = want[n]; hok = g['h'] == H[n]
    ok = hok and g['sr'] == [len(e)] and len(g['sv']) == len(e) and all(g['sv'][i] == e[i] for i in range(1, len(e)))
    if g['sv']: cyc[n] = g['sv'][0]
    print(f'{n}: img {"OK" if hok else "MISMATCH"} bs {g["bs"]} SR {g["sr"]} SV {" ".join(hex(x) for x in g["sv"])}'
          f'  {"EXACT FIELDS OK" if ok else "OFF"}  cycles {g["sv"][0] if g["sv"] else None} (sim {e[0]})'
          + (f'  {"IN BAND" if abs(g["sv"][0] - e[0]) <= 0.10 * e[0] else "OUT OF BAND"} ({100 * (g["sv"][0] - e[0]) / e[0]:+.1f} %)' if g['sv'] else ''))
    if not ok: bad.append(n)
    if not g['bs'] or RES[ARM] not in g['bs'][-1]: wrongbs.append(n)
v = 'NO DATA' if nodata else ('WRONG BITSTREAM' if wrongbs else ('FINDING' if bad else 'EXACT'))
patt = []
if v == 'EXACT':
    p = [cyc[f'slot-{i}'] for i in (1, 2, 3, 4)]
    near = lambda a, b: abs(a - b) <= 0.02 * min(a, b)
    if ARM == 'dyntrim':
        p1 = near(p[0], p[3]) and near(p[1], p[2]) and p[1] < 0.95 * p[0]
        print(f'P1d pair cycles {p}: {"HOLDS" if p1 else "FAILS"}  (1~4, 2~3 within 2 %, 2 > 5 % below 1)')
    else:
        p1 = (max(p) - min(p)) <= 0.02 * min(p)
        print(f'P1 pair cycles {p}: {"HOLDS" if p1 else "FAILS"}  (same within 2 %)')
    p2 = cyc['slot-5'] > cyc['slot-6']
    print(f'P2 plain same-slot {cyc["slot-5"]} > other-slot {cyc["slot-6"]}: {"HOLDS" if p2 else "FAILS"}')
    patt = [p1, p2]
    L = cyc['slot-9']
    per = lambda n, k: (cyc[n] - L) / (N * k)
    print(f'PER OP over the bare loop (cycles): ldc {per("slot-7", 1):.2f}  stc {per("slot-8", 1):.2f}  '
          f'stc+ldc same slot {per("slot-1", 1):.2f}  other slot {per("slot-2", 1):.2f}  '
          f'sd+ld same slot {per("slot-5", 1):.2f}  other slot {per("slot-6", 1):.2f}  (loop {L / N:.2f}/iter)')
    v = 'AS EXPECTED' if all(patt) else 'PATTERN FAILS'
oob = [n for n in cyc if abs(cyc[n] - want[n][0]) > 0.10 * want[n][0]]
print(f'CYCLE BAND (sim +-10 %, reported): {"ALL IN BAND" if not oob and cyc else ("NO CYCLES" if not cyc else "OUT OF BAND " + " ".join(oob))}')
print(f'ARM {ARM}: {v}  (off {bad} wrong-bitstream {wrongbs} no-data {nodata})')
sys.exit(2 if v == 'NO DATA' else (0 if v == 'AS EXPECTED' else 1))
