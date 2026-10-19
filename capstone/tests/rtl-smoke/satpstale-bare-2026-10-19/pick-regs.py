#!/usr/bin/env python3
"""Pick three GPRs a test never names (code lines with comments stripped, plus any test-local include), for the board
wrapper's recorder scratch (BOARD_R_CAP, BOARD_R_T1, BOARD_R_T2). Prints '-DBOARD_R_CAP=xA -DBOARD_R_T1=xB -DBOARD_R_T2=xC'."""
import re, sys, os
ABI = {'zero':0,'ra':1,'sp':2,'gp':3,'tp':4,'t0':5,'t1':6,'t2':7,'s0':8,'fp':8,'s1':9,'a0':10,'a1':11,'a2':12,'a3':13,'a4':14,
       'a5':15,'a6':16,'a7':17,'s2':18,'s3':19,'s4':20,'s5':21,'s6':22,'s7':23,'s8':24,'s9':25,'s10':26,'s11':27,'t3':28,'t4':29,
       't5':30,'t6':31}
def text(path, seen):
    out = []
    for l in open(path):
        m = re.match(r'\s*#include\s+"([^"]+)"', l)
        if m:
            p = os.path.join(os.path.dirname(path), m.group(1))
            if os.path.isfile(p) and p not in seen and '*' not in seen: seen.add(p); out.append(text(p, seen))
            continue
        l = re.sub(r'//.*|/\*.*?\*/', '', l)
        if re.match(r'\s*#\s', l) or re.match(r'\s*#\s*$', l): continue    # '# comment' lines (cpp passes them; asm comment)
        out.append(l.split('#')[0] if not l.lstrip().startswith('#') else l)
    return '\n'.join(out)
NOINC = '--no-includes' in sys.argv; sys.argv = [a for a in sys.argv if a != '--no-includes']
t = text(sys.argv[1], set() if not NOINC else {'*'})
used = set()
for tok in re.findall(r'\b(x[0-9]{1,2}|zero|ra|sp|gp|tp|fp|t[0-6]|s[0-9]|s1[01]|a[0-7])\b', t):
    used.add(int(tok[1:]) if re.fullmatch(r'x[0-9]+', tok) else ABI[tok])
# The wrapper's CAP_PASS uses x23..x31 only after the test is over; avoid x0..x4 (zero, ra, sp, gp, tp).
free = [r for r in [16, 17, 8, 4, 2, 9, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 5, 6, 7, 10, 11, 12, 13, 14, 15, 25]
        if r not in used and r not in (0, 1, 3)]
if len(free) < 3: sys.exit(f"ERROR: fewer than three free GPRs in {sys.argv[1]}: used {sorted(used)}")
print(f"-DBOARD_R_CAP=x{free[0]} -DBOARD_R_T1=x{free[1]} -DBOARD_R_T2=x{free[2]}")
