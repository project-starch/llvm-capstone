#!/usr/bin/env python3
"""inline-io-funcs.py RUNDIR MODE N [direct] [--top K]: per-call attempts by function, the difference
of the two windows of one (mode, N, direct) at its lowest and highest round counts (lane AB)."""
import glob, json, os, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
if "--top" in sys.argv:
    args.remove(str(top))
rundir, mode, n = args[0], args[1], int(args[2])
direct = len(args) > 3 and args[3] == "direct"
wins = {}
for f in glob.glob(os.path.join(rundir, "invocations", "[0-9]" * 4 + ".json")):
    d = json.load(open(f))
    a = d["argv"]
    if a[:2] == [mode, str(n)] and (len(a) > 3 and a[3] == "direct") == direct and d["detail"]["window"] is not None:
        wins[int(a[2])] = d["detail"]["window"]
if len(wins) < 2:
    sys.exit(f"ERROR: fewer than two windows for {mode} {n} direct={direct} in {rundir}")
r1, r2 = min(wins), max(wins)
def funcs(w):
    out = {}
    for cls, fn, v in json.load(open(os.path.join(rundir, "profile", f"window-{w}.json")))["all_functions"]:
        out[(cls, fn)] = out.get((cls, fn), 0) + v
    return out
a, b = funcs(wins[r1]), funcs(wins[r2])
rows = sorted(((b.get(k, 0) - a.get(k, 0)) / (r2 - r1), k) for k in set(a) | set(b))
rows.reverse()
print(f"# {rundir} {mode} N={n} direct={int(direct)}: per call, rounds {r1}->{r2}")
for v, (cls, fn) in rows[:top]:
    print(f"{v:10,.0f}  {cls:9s} {fn}")
