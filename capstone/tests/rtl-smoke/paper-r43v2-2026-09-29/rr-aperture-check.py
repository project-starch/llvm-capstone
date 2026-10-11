#!/usr/bin/env python3
"""For each [wedge] refusal-record line (sw=204..209) in a driver.log, report the switch value actually set when the
byte was taken: the last switch_state echo before that line, since the previous [wedge] line. A record byte is
DATA only when that value equals the labelled aperture. Exit 1 if any rr line was read at the wrong aperture, 2 if
none was found."""
import re, sys, ast
L = open(sys.argv[1], errors='replace').read().split('\n')
def sv(l):
    m = re.search(r"'states': (\[[01, ]+\])", l)
    return None if not m else sum(1 << i for i, x in enumerate(ast.literal_eval(m.group(1))) if x)
wl = [k for k, l in enumerate(L) if '[wedge] sw=' in l]
rr = [k for k in wl if re.search(r'\[wedge\] sw=20[4-9] ', L[k])]
if not rr: print("no refusal-record lines found"); sys.exit(2)
bad = 0
for i in rr:
    target = int(re.search(r'sw=(\d+)', L[i]).group(1))
    prev = max([k for k in wl if k < i], default=0)
    sws = [sv(l) for l in L[prev:i] if 'switch_state' in l and sv(l) is not None]
    actual = sws[-1] if sws else None
    v = re.search(r'0x([0-9a-f]{2}) ', L[i])
    ok = actual == target
    bad += (not ok)
    print(f"sw={target} value={('0x'+v.group(1)) if v else 'UNREAD'} read-at={actual} {'OK' if ok else 'WRONG APERTURE -> not data'}")
sys.exit(1 if bad else 0)
