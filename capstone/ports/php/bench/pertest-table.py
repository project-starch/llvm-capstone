#!/usr/bin/env python3
"""Per-test Capstone/plain attempt ratios from two profile-php.sh runs of the per-test scripts.

usage: pertest-table.py PLAIN_RUN CAPSTONE_RUN
Windows are matched by the script each invocation ran (its argv); a script present in one run only,
or a run without its profile/window-N.json, is an error, not a missing row. Columns: all attempts in
the window, then the program class alone (the PHP image), each as plain, Capstone and the ratio.
"""
import glob, json, os, sys


def windows(run):
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "invocations", "[0-9]" * 4 + ".json"))):
        d = json.load(open(f))
        if d.get("route") not in ("guest", "guest-plain"):
            continue
        n = d["detail"]["window"]
        p = os.path.join(run, "profile", f"window-{n}.json")
        if not os.path.exists(p):
            sys.exit(f"pertest-table: {p} missing")
        w = json.load(open(p))
        out[d["argv"][-1]] = (w["total_attempts"], w["classes"].get("program", 0))
    if not out:
        sys.exit(f"pertest-table: no guest invocations under {run}")
    return out


if len(sys.argv) != 3:
    sys.exit(__doc__)
a, b = windows(sys.argv[1]), windows(sys.argv[2])
if set(a) != set(b):
    sys.exit(f"pertest-table: scripts differ: only plain {sorted(set(a) - set(b))}, only capstone {sorted(set(b) - set(a))}")
print(f"{'test':22s} {'plain':>15s} {'capstone':>15s} {'ratio':>6s}   {'plain prog':>15s} {'cap prog':>15s} {'ratio':>6s}")
ta = tb = pa = pb = 0
for s in sorted(a):
    (x, px), (y, py) = a[s], b[s]
    ta += x; tb += y; pa += px; pb += py
    print(f"{s[:-4]:22s} {x:15,} {y:15,} {y / x:6.3f}   {px:15,} {py:15,} {py / px:6.3f}")
print(f"{'all':22s} {ta:15,} {tb:15,} {tb / ta:6.3f}   {pa:15,} {pb:15,} {pb / pa:6.3f}")
