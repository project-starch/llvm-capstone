#!/usr/bin/env python3
"""Compare two tools/profile.sh runs function by function: the same arm twice (determinism), or
two arms of one benchmark (where the instructions go).

usage: prof-compare.py A B [--top N] [--json OUT]
  A, B   run directories (OUT of tools/profile.sh) or their profile.json files

Prints, for every class (program, kernel, module, launcher, profrun, firmware, other...), B minus A
in attempts and the functions whose counts differ most, both signs, with the residual of functions
present in only one side. Functions are matched by (class, name); a renamed, cloned or newly
inlined function therefore shows up as an unmatched pair, which the residual lines make visible
rather than hiding. The sum of every class line equals the total delta.
"""
import argparse, collections, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--json")
A = ap.parse_args()


def load(p):
    if os.path.isdir(p):
        p = os.path.join(p, "profile.json")
    d = json.load(open(p))
    funcs = {(c, f): v for c, f, v in d["all_functions"]}
    return d, funcs


da, fa = load(A.a)
db, fb = load(A.b)
# Two work-bounded slices (tools/stop-at.py) compare only when they stopped at the same entry of the
# same function: then both did the same work. A slice and a full run, or two slices of different
# work, are refused.
def work(d):
    s = d.get("slice")
    return None if not s else (s["symbol"], s["count"], json.dumps(s.get("start"), sort_keys=True))
if work(da) != work(db):
    sys.exit(f"prof-compare: the runs did not do the same work: A {work(da) or 'full run'}, B {work(db) or 'full run'}")
if work(da):
    print(f"slices: both stopped at entry {da['slice']['count']:,} of {da['slice']['symbol']}"
          + (f", started at {da['slice']['start']}" if da['slice'].get('start') else ""))
ta, tb = da["total_attempts"], db["total_attempts"]
print(f"A {A.a}: {ta:,} attempts ({len(da['windows'])} windows)")
print(f"B {A.b}: {tb:,} attempts ({len(db['windows'])} windows)")
print(f"B - A = {tb - ta:+,} ({(tb / ta - 1) * 100 if ta else float('nan'):+.3f}%)")
ia, ib = da["classes"].get("kernel idle", 0), db["classes"].get("kernel idle", 0)
print(f"without the idle loop: A {ta - ia:,}  B {tb - ib:,}  B - A = {(tb - ib) - (ta - ia):+,} ({((tb - ib) / (ta - ia) - 1) * 100 if ta - ia else float('nan'):+.3f}%)")
classes = sorted(set(da["classes"]) | set(db["classes"]), key=lambda c: -(db["classes"].get(c, 0) + da["classes"].get(c, 0)))
out = {"a": A.a, "b": A.b, "total_a": ta, "total_b": tb, "classes": {}}
print(f"\n{'class':26s} {'A':>16s} {'B':>16s} {'B-A':>16s}")
for c in classes:
    va, vb = da["classes"].get(c, 0), db["classes"].get(c, 0)
    print(f"{c:26s} {va:16,} {vb:16,} {vb - va:+16,}")
    out["classes"][c] = {"a": va, "b": vb, "delta": vb - va}
keys = set(fa) | set(fb)
deltas = {k: fb.get(k, 0) - fa.get(k, 0) for k in keys}
only_a = {k: fa[k] for k in fa if k not in fb}
only_b = {k: fb[k] for k in fb if k not in fa}
print(f"\nfunctions: {len(keys)} ({len(only_a)} only in A: {sum(only_a.values()):,} attempts; "
      f"{len(only_b)} only in B: {sum(only_b.values()):,} attempts)")
same = sum(1 for k in keys if deltas[k] == 0)
print(f"identical counts: {same} of {len(keys)} functions")
ranked = sorted(keys, key=lambda k: -abs(deltas[k]))
print(f"\n{'B-A':>16s} {'A':>14s} {'B':>14s}  class      function")
for k in ranked[:A.top]:
    if deltas[k] == 0:
        break
    tag = " (only A)" if k in only_a else " (only B)" if k in only_b else ""
    print(f"{deltas[k]:+16,} {fa.get(k, 0):14,} {fb.get(k, 0):14,}  {k[0]:10s} {k[1]}{tag}")
out["top"] = [[k[0], k[1], fa.get(k, 0), fb.get(k, 0), deltas[k]] for k in ranked[:A.top * 4]]
out["only_a"] = sum(only_a.values()); out["only_b"] = sum(only_b.values())
out["identical_functions"] = same; out["functions"] = len(keys)
if A.json:
    json.dump(out, open(A.json, "w"), indent=1)
