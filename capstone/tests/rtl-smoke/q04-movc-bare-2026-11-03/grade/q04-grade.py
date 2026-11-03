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
src = 'UART'
# CORRECTED 2026-11-03 after board 1's reading. This image never reaches the board wrapper's CAP_PASS (the test ends
# with its own tohost store and `j 1b`), so its UART carries no SR/SV lines on silicon; the readings are in the
# recorder area, which is where PREREG.md's vectors were simulated from. When the UART has none, read the recorder
# from the run's own gdb.txt (`x/1gx 0x80001140` = the count, then `x/Ngx 0x80001148`). Was: UART only, and an empty
# UART graded FINDING instead of NO DATA.
REC = 0x80001140
if not sr and (D / 'gdb.txt').exists():
    g = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', (D / 'gdb.txt').read_text(errors='replace'))
    words = {}
    for line in g.split('\n'):
        m2 = re.match(r'\s*0x([0-9a-f]+):\s+((?:0x[0-9a-f]+\s*)+)$', line.strip())
        if m2:
            a = int(m2.group(1), 16)
            for k, w in enumerate(m2.group(2).split()): words[a + 8 * k] = int(w, 16)
    if REC in words:
        n = words[REC]; sr = [n]; sv = [words.get(REC + 8 * (i + 1)) for i in range(n)]
        src = 'RECORDER (gdb.txt)'
        if None in sv: sv = []
if not sr: print(f'NO DATA: no SR line in the UART and no recorder dump at {REC:#x} in gdb.txt ({D})'); sys.exit(2)
m = re.search(r'image (\S+\.bin) ->', lg); img = pathlib.Path(m.group(1)) if m else None
h = hashlib.sha256(img.read_bytes()).hexdigest() if img and img.exists() else None
bs = re.findall(r"resident NV bitstream = '([^']+)'", lg)
print(f'source {src}; img {"OK" if h == H else "MISMATCH"} bs {bs} SR {sr} SV {" ".join(hex(x) for x in sv)}')
if h != H: v = 'IMAGE MISMATCH'
elif not bs or RES[ARM] not in bs[-1]: v = 'WRONG BITSTREAM'
elif sr == [26] and sv == PRESENT: v = 'PRESENT'
elif sr == [26] and sv == ABSENT: v = 'ABSENT'
else: v = 'FINDING'
print(f'ARM {ARM}: Q-04 (b) {v}')
sys.exit(0 if v == 'PRESENT' else 1)
