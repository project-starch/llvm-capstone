#!/usr/bin/env python3
"""Turn a PoisonCap run of this corpus into the committed result record:
matrix.tsv (one line per arm, mode 0 and 1) and inputs.json (binaries,
platform fingerprint, controls) beside a hand-written README."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path)
p.add_argument("results", type=Path)
p.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent.parent)
a = p.parse_args()
v = json.loads((a.run / "verdicts.json").read_text())
cases = {}
for d in sorted(a.corpus.glob("[0-9][0-9]_*")):
    c = json.loads((d / "case.json").read_text())
    cases[c["case"]] = (c["upstream_fix"], d.name.split("_", 2)[2])
columns = "case fix name mode expected passed exit fault_pc expect_pc fault_at_probe completed counters".split()
lines = ["\t".join(columns)]
for arm in v["arms"]:
    fix, name = cases[arm["case"]]
    first = arm["faults"][0] if arm["faults"] else {}
    r = arm.get("report") or {}
    lines.append("\t".join("" if x is None else str(x) for x in (
        arm["case"], fix, name, arm["mode"], arm["expected"], arm["passed"], arm["exit"], first.get("pc", ""),
        arm.get("expect"), arm.get("fault_at_probe", ""), r.get("completed", ""), arm.get("counters") or "")))
inputs = {k: v[k] for k in ("runtime_revocation", "platform", "binaries", "platform_controls")}
inputs["arms_total"] = len(v["arms"])
inputs["arms_passed"] = sum(arm["passed"] for arm in v["arms"])
a.results.mkdir(parents=True, exist_ok=False)
(a.results / "matrix.tsv").write_text("\n".join(lines) + "\n")
(a.results / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
print(f"{inputs['arms_passed']}/{inputs['arms_total']} arms passed; wrote {a.results}")
