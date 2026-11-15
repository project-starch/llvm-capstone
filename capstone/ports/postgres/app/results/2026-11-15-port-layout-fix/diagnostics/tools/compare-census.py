#!/usr/bin/env python3
"""compare-census.py STOCK_RUN PROTECTED_RUN [RUN5_PROTECTED_BLOCKS]: the block census of two runs.

Each RUN is a run-pgbench-reuse.py work directory of a census build. Prints both EXP-BLOCKS-END
lines, the observer counters, every (path, context) row that differs, the per-path totals, the
EXP-SLACK line of the protected run and the reusediff of the two runs' bins.
"""
import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"^EXP-BLOCKS path=(\S+) name=(.*?) blocks=(\d+) bytes=(\d+)$", re.M)


def load(run):
    run = Path(run)
    rep = json.loads((run / "rep.json").read_text())
    out = json.loads((run / "results.json").read_text())[0]["output"].replace("\r", "")
    rows = {(p, n): (int(b), int(s)) for p, n, b, s in LINE.findall(out)}
    end = re.search(r"^EXP-BLOCKS-END .*$", out, re.M)
    slack = re.search(r"^EXP-SLACK .*$", out, re.M)
    return rep, rows, end.group(0) if end else "NO END", slack.group(0) if slack else None, out


def rd(a, b):
    na, nb = sum(a), sum(b)
    return sum(abs(x / na - y / nb) for x, y in zip(a, b)) / 2


(srep, s, send, _, _), (prep, p, pend, pslack, _) = load(sys.argv[1]), load(sys.argv[2])
for name, rep, end in (("stock", srep, send), ("protected", prep, pend)):
    g = rep["reuse_gap"]["contexts"]
    print(f"{name:9s} {end} rc={rep['rc']} rows_equal_native={rep.get('rows_equal_native')} guest_s={rep['guest_seconds']}",
          {k: g[k] for k in ("issues", "releases", "reuses", "distinct_starts", "error")})
print("protected", pslack)
print("reusediff stock/protected", round(rd(srep["reuse_gap"]["contexts"]["bins"], prep["reuse_gap"]["contexts"]["bins"]), 6))
paths = sorted({k[0] for k in s} | {k[0] for k in p})
print("\nper path: blocks (bytes), stock then protected")
for path in paths:
    sb = sum(v[0] for k, v in s.items() if k[0] == path); ss = sum(v[1] for k, v in s.items() if k[0] == path)
    pb = sum(v[0] for k, v in p.items() if k[0] == path); ps = sum(v[1] for k, v in p.items() if k[0] == path)
    print(f"{path:20s} {sb:>8} ({ss:>11})  {pb:>8} ({ps:>11})")
print(f"{'total':20s} {sum(v[0] for v in s.values()):>8}  {' ' * 13} {sum(v[0] for v in p.values()):>8}")
diff = sorted(k for k in set(s) | set(p) if s.get(k) != p.get(k))
print(f"\nrows: stock {len(s)}, protected {len(p)}, differing {len(diff)}")
for k in diff:
    print("  DIFF", k, s.get(k), p.get(k))
for k in (("aset-new", "MessageContext"), ("aset-new", "index info")):
    print("  ", k, "stock", s.get(k), "protected", p.get(k))
if len(sys.argv) > 3:
    run5 = {(a, b): (int(c), int(d)) for a, b, c, d in LINE.findall(Path(sys.argv[3]).read_text())}
    print("\nrun-5 stock census against this stock census: differing rows",
          sum(1 for k in set(run5) | set(s) if run5.get(k) != s.get(k)))
