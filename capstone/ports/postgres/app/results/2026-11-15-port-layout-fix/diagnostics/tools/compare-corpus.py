#!/usr/bin/env python3
"""compare-corpus.py: every sql-repros virtual-pg-pools case, run-4 against run-6 (this fix).

Run-4's reading of a case is its main bundle's, unless that is NO-READING:infra and a rerun bundle
(07, 10-11) has it; run-6 likewise with its two bundles."""
import json
from pathlib import Path

E = Path.home() / "asr-work/eval"


def rows(*bundles):
    got = {}
    for b in bundles:
        p = E / b / "verdicts.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            key = r["observation"]["case"]
            v = r["verdict"] + (":" + r["reason"] if r.get("reason") else "")
            if key in got and not got[key][0].startswith("NO-READING:infra") and v.startswith("NO-READING:infra"):
                continue
            got[key] = (v, r.get("evidence", "")[:110])
    return got


r4 = rows("run-4/postgres-sql/virtual-pg-pools", "run-4/postgres-sql-07/virtual-pg-pools",
          "run-4/postgres-sql-10-11/virtual-pg-pools")
r6 = rows("run-6/corpora/postgres-sql-virtual-pg-pools", "run-6/corpora/postgres-sql-10-11-virtual-pg-pools")
for case in sorted(set(r4) | set(r6), key=str):
    a, b = r4.get(case, ("-", "")), r6.get(case, ("-", ""))
    print(f"{'SAME' if a[0] == b[0] else 'CHANGED':7s} {str(case)[:52]:52s} run-4 {a[0]:24s} run-6 {b[0]}")
    print(f"        run-4: {a[1]}\n        run-6: {b[1]}")
for name, r in (("run-4", r4), ("run-6", r6)):
    print(name, {v: sum(1 for x in r.values() if x[0] == v) for v in sorted({x[0] for x in r.values()})})
