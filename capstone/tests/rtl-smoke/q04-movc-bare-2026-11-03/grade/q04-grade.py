#!/usr/bin/env python3
"""q04-grade.py <run dir> <bs3|objb>: grade the Q-04 (b) bare arm against q04-movc-bare-2026-11-03/PREREG.md.
Reads <run dir>/movc-untagged/{uart.txt, run.log}: the SR/SV lines, the image that run loaded (hashed here), and the
resident bitstream name, which must name the arm (e7af48457 / 603528b8b), else WRONG BITSTREAM.
- PRESENT: the vector equals the Q-04 (b) vector (simulated on bitstream 3's tokens and on ObjBounds, identical).
- ABSENT: it equals the field-RTL vector (c7e6f604f: the untagged source nulled, TESTNUM 1).
- FINDING: anything else.
Exit 0 = PRESENT; 1 = any other verdict; 2 = NO DATA."""
import sys, re, hashlib, pathlib
D = pathlib.Path(sys.argv[1]) / 'movc-untagged'; ARM = sys.argv[2]
if ARM not in ('bs3', 'objb'): sys.exit('arm must be bs3 or objb')
H = '960e6b7992bbeb2c653c02a03fbf415795294051a0b492390d2864dba35f15d7'
RES = {'bs3': 'e7af48457', 'objb': '603528b8b'}
V = lambda s: [int(x, 16) for x in s.split()]
TAG = '0x0 0x80001000 0x0 0x80001000 0x0 0x80001000 0x0 0x80001000 0x0 0x80001000 0x0 0x80001000 0x80001000 0x0 0x80001000 0x0'
PRESENT = V(TAG + ' 0x1122334455667788 0x1122334455667788 0x1122334455667788 0x0 0x0 0x55 0x77 0x0 0x0 0x600d')
ABSENT = V(TAG + ' 0x0 0x0 0x1122334455667788 0x0 0x0 0x55 0x0 0x0 0x0 0x1')
u, r = D / 'uart.txt', D / 'run.log'
if not u.exists() or not r.exists(): print(f'NO DATA: {D}'); sys.exit(2)
t = u.read_text(errors='replace'); lg = r.read_text(errors='replace')
sr = [int(x, 16) for x in re.findall(r'^SR ([0-9A-Fa-f]{16})\s*$', t, re.M)]
sv = [int(x, 16) for x in re.findall(r'^SV ([0-9A-Fa-f]{16})\s*$', t, re.M)]
m = re.search(r'image (\S+\.bin) ->', lg); img = pathlib.Path(m.group(1)) if m else None
h = hashlib.sha256(img.read_bytes()).hexdigest() if img and img.exists() else None
bs = re.findall(r"resident NV bitstream = '([^']+)'", lg)
print(f'img {"OK" if h == H else "MISMATCH"} bs {bs} SR {sr} SV {" ".join(hex(x) for x in sv)}')
if h != H: v = 'IMAGE MISMATCH'
elif not bs or RES[ARM] not in bs[-1]: v = 'WRONG BITSTREAM'
elif sr == [26] and sv == PRESENT: v = 'PRESENT'
elif sr == [26] and sv == ABSENT: v = 'ABSENT'
else: v = 'FINDING'
print(f'ARM {ARM}: Q-04 (b) {v}')
sys.exit(0 if v == 'PRESENT' else 1)
