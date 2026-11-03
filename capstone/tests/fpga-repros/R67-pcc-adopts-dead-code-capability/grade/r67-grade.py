#!/usr/bin/env python3
"""r67-grade.py <run dir> <bs3|objb|fixed>: grade the R-67 kit against R67-pcc-adopts-dead-code-capability/PREREG.md.
Reads <run dir>/<image>/{uart.txt, run.log} for t1-norevoke, t1-untagged, t1-oob and t1-fire: the SR/SV lines, the
image that run loaded (hashed here) and the resident bitstream name, which must name the arm (e7af48457, 603528b8b, or
for 'fixed' any name given with --bitstream). Readings R1..R11 as in tests/lv-t1-body.h.
- The three controls must read their pre-registered vectors exactly (they are the same on every bitstream): else VOID.
- t1-fire: HOLE = the target ran (R4 0, R6 MARK, R7 0 traps); FIXED = R4 0, R6 0, at least one trap, cause 25 or 1,
  R9 (mepc - tgt) 0; anything else is FINDING.
Exit 0 = the arm's pre-registered verdict (HOLE for bs3/objb, FIXED for fixed); 1 = any other verdict; 2 = NO DATA."""
import sys, re, hashlib, pathlib
D = pathlib.Path(sys.argv[1]); ARM = sys.argv[2]
RES = {'bs3': 'e7af48457', 'objb': '603528b8b'}
if ARM not in ('bs3', 'objb', 'fixed'): sys.exit('arm must be bs3, objb or fixed')
WANT_BS = RES.get(ARM) or (sys.argv[sys.argv.index('--bitstream') + 1] if '--bitstream' in sys.argv else None)
H = {'t1-norevoke': 'ba89db3f915a53b5fe4f7ebee3fd2880a4ffbf838c8a5eb5431e781ea0e4a1e5',
     't1-untagged': '92aea79fa39a020614febc597ca8c21ac5ec8aaa0b951fdaee7a4b14ec263d9e',
     't1-oob':      '60d0db28ee338fd0c752e72f02fd5bef4629ab8591b444c990ca0135d43b7439',
     't1-fire':     '6482059c119290e9edede3337760085d7344478840ef80edb7313d6434be1f92'}
V = lambda s: [int(x, 16) for x in s.split()]
CTL = {'t1-norevoke': V('0x7101 0x800009e0 0x1 0x1 0x0 0x4d41524b 0x0 0x0 0xffffffff7ffff620 0x0 0x59'),
       't1-untagged': V('0x7101 0x800009f0 0x1 0x1 0x0 0x0 0x1 0x18 0xfffffffffffffb10 0x800009f0 0x59'),
       't1-oob':      V('0x7101 0x800009e0 0x1 0x1 0x0 0x0 0x1 0x1c 0x10 0x800009f0 0x59')}
HOLE = V('0x7101 0x800009e0 0x1 0x0 0x0 0x4d41524b 0x0 0x0 0xffffffff7ffff620 0x0 0x59')
MARK = 0x4d41524b
def read(n):
    u, r = D / n / 'uart.txt', D / n / 'run.log'
    if not u.exists() or not r.exists(): return None
    t = u.read_text(errors='replace'); lg = r.read_text(errors='replace')
    sr = [int(x, 16) for x in re.findall(r'^SR ([0-9A-Fa-f]{16})\s*$', t, re.M)]
    sv = [int(x, 16) for x in re.findall(r'^SV ([0-9A-Fa-f]{16})\s*$', t, re.M)]
    m = re.search(r'image (\S+\.bin) ->', lg); img = pathlib.Path(m.group(1)) if m else None
    h = hashlib.sha256(img.read_bytes()).hexdigest() if img and img.exists() else None
    bs = re.findall(r"resident NV bitstream = '([^']+)'", lg)
    return dict(sr=sr, sv=sv, h=h, bs=bs)
g = {n: read(n) for n in H}
if any(v is None for v in g.values()):
    print('NO DATA:', [n for n, v in g.items() if v is None]); sys.exit(2)
bad = []
for n, x in g.items():
    ok_img = x['h'] == H[n]
    ok_bs = bool(x['bs']) and WANT_BS is not None and WANT_BS in x['bs'][-1]
    print(f'{n}: img {"OK" if ok_img else "MISMATCH"} bs {x["bs"]} SR {x["sr"]} SV {" ".join(hex(v) for v in x["sv"])}')
    if not ok_img or not ok_bs: bad.append(n)
if bad: print(f'ARM {ARM}: IMAGE OR BITSTREAM MISMATCH {bad}'); sys.exit(1)
void = [n for n in CTL if not (g[n]['sr'] == [11] and g[n]['sv'] == CTL[n])]
if void: print(f'ARM {ARM}: VOID (a control is off: {void})'); sys.exit(1)
f = g['t1-fire']['sv']
if g['t1-fire']['sr'] == [11] and f == HOLE: v = 'HOLE'
elif g['t1-fire']['sr'] == [11] and len(f) == 11 and f[3] == 0 and f[5] == 0 and f[6] >= 1 and f[7] in (25, 1) and f[8] == 0: v = 'FIXED'
else: v = 'FINDING'
print(f'ARM {ARM}: t1-fire {v}')
sys.exit(0 if v == ('FIXED' if ARM == 'fixed' else 'HOLE') else 1)
