#!/usr/bin/env python3
"""objb-wgrade.py <image dir (uart.txt + run.log)>: grade objb-stale-w against PREREG addendum 2 (land/rtl-objb).
Hashes the image the run.log names, reads SR/SV and the resident bitstream. Exit 0 = WITNESSED; 1 = VOID / STOP /
FINDING; 2 = NO DATA."""
import sys, re, hashlib, pathlib
D = pathlib.Path(sys.argv[1])
H = 'ae0e1cb35dcc42d3c7bf5e370c6b1ec68a1ed1b9d2305b0a9e9f94191357e582'
V = lambda s: [int(x, 16) for x in s.split()]
OBJB = V('0x0 0x0 0x0 0x0 0x0 0x0 0x0 0x3333 0x0 0x0 0x19 0x19 0x19 0x2222 0x3')
BS3 = V('0x19 0x19 0x19 0x19 0x0 0x19 0x19 0x77 0x6 0x0 0x19 0x19 0x19 0x0 0x9')
BN = 'caplifive_vp2c_603528b8b.bit'
u, r = D / 'uart.txt', D / 'run.log'
if not u.exists() or not r.exists():
    print(f'NO DATA: {u} / {r}'); sys.exit(2)
t, lg = u.read_text(errors='replace'), r.read_text(errors='replace')
sr = [int(x, 16) for x in re.findall(r'^SR ([0-9A-Fa-f]{16})\s*$', t, re.M)]
sv = [int(x, 16) for x in re.findall(r'^SV ([0-9A-Fa-f]{16})\s*$', t, re.M)]
m = re.search(r'image (\S+\.bin) ->', lg); img = pathlib.Path(m.group(1)) if m else None
h = hashlib.sha256(img.read_bytes()).hexdigest() if img and img.exists() else None
bs = re.findall(r"resident NV bitstream = '([^']+)'", lg)
print(f'image {img} sha256 {"OK" if h == H else "MISMATCH " + str(h)}; resident {bs}; SR {sr}; SV {" ".join(hex(x) for x in sv)}')
if h != H or not bs or any(b != BN for b in bs):
    v = 'VOID (wrong image or resident bitstream)'
elif sr != [15] or len(sv) != 15:
    v = 'VOID (no return or SR is not 15)'
elif sv[9] != 0:
    v = 'VOID (position 9, the live-node control, is not 0)'
elif sv[:9] == BS3[:9]:
    v = 'STOP: wrong bitstream (positions 0-8 read bitstream 3)'
elif sv == OBJB:
    v = 'WITNESSED'
else:
    d = [i for i in range(15) if sv[i] != OBJB[i]]
    live = [i for i in (10, 11, 12) if sv[i] == 0]
    v = f'FINDING at positions {d}' + (f'; node LIVE on ObjBounds at {live}' if live else '')
print('VERDICT', v)
sys.exit(0 if v == 'WITNESSED' else 1)
