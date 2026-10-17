#!/usr/bin/env python3
# compare-bootp.py <boot P board txt> <boot BL board txt> : rung retvals, mruby hashes, bench TIME lines and monitor
# report tags, side by side; exit 2 if either file has no RESULT line ("no data" is an error).
import re, sys
def read(p):
    t = open(p, errors='replace').read()
    r = dict(re.findall(r'RESULT (\S+) retval=(-?\d+)', t))
    m = {k: (sha, secs) for k, sha, secs in re.findall(r'MRUBY (\S+) sha ([0-9a-f]+) bytes \d+ want [0-9a-f]+ rc \d+ secs ([0-9.]+)', t)}
    tm = dict((k, int(v)) for k, v in re.findall(r'TIME (\S+) (\d+) ms', t))
    tags = {k: len(re.findall(k, t)) for k in ('EXCT', 'EXCS', 'RGNF', 'ILLX')}
    ticks = re.findall(r'launch ticks image=(\d+) hash=\d+ spawner=\d+ device=\d+ domain=(\d+) regions=(\d+) seccomp=\d+ total=(\d+)', t)
    return r, m, tm, tags, ticks
P, B = read(sys.argv[1]), read(sys.argv[2])
if not P[0] or not B[0]: print("NO DATA"); sys.exit(2)
print("rung retvals:", "IDENTICAL" if P[0] == B[0] else "DIFFER", B[0])
for k in P[1]: print(f"mruby {k}: P {P[1][k]}  BL {B[1].get(k)}  sha {'same' if B[1].get(k, ('',))[0] == P[1][k][0] else 'DIFFERENT'}")
for k in P[2]: print(f"TIME {k}: P {P[2][k]} BL {B[2].get(k)} ms  {100*(B[2].get(k, 0)-P[2][k])/P[2][k]:+.2f} %")
print("tags BL:", B[3], " P:", P[3])
for i, (p, b) in enumerate(zip(P[4], B[4])): print(f"launch {i}: total P {int(p[3])/12.5e6:.2f}s BL {int(b[3])/12.5e6:.2f}s  domain P {int(p[1])/12.5e6:.2f} BL {int(b[1])/12.5e6:.2f}  regions P {int(p[2])/12.5e6:.2f} BL {int(b[2])/12.5e6:.2f}")
