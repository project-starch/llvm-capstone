#!/usr/bin/env python3
"""Turn a CheriBSD run of this corpus into the committed result record.

Reads runners/cheribsd/run-defects.py's verdicts.json and the guest summary,
writes matrix.tsv (one line per arm, its control beside it) and inputs.json
(binaries, platform fingerprint, ABI-control line) into a fresh results/<stamp>/
directory beside a hand-written README. Raw guest logs stay outside the repository.
"""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path, help="the runner's output directory")
p.add_argument("results", type=Path, help="results/<stamp> directory to create")
p.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent.parent)
a = p.parse_args()
v = json.loads((a.run / "verdicts.json").read_text())
summary = json.loads((a.run / "guest/summary.json").read_text())
cases = {}
for d in sorted(a.corpus.glob("[0-9][0-9]_*")):
    c = json.loads((d / "case.json").read_text())
    cases[c["case"]] = (c["upstream_fix"], d.name.split("_", 2)[2])
control = v["revocation_control"]
columns = "case fix name revocation expected passed exit completed nodes node_reuses control_expected control_passed control_fault_pc control_expect".split()
lines = ["\t".join(columns)]
for arm in v["arms"]:
    fix, name = cases[arm["case"]]
    r = arm.get("report") or {}
    lines.append("\t".join(str(x) for x in (
        arm["case"], fix, name, v["runtime_revocation"], arm["expected"], arm["passed"], arm["exit"],
        r.get("completed"), r.get("nodes"), r.get("node_reuses"), control["expected"], control["passed"],
        (control.get("fault") or {}).get("pc", ""), control.get("expect"))))
inputs = {
    "runtime_revocation": v["runtime_revocation"],
    "guest_default_revocation": v["guest_default_revocation"],
    "platform": v["platform"],
    "binaries": v["binaries"],
    "platform_controls": [dict(name=c["name"], passed=c["passed"], marker=c.get("marker")) for c in v["platform_controls"]],
    "revocation_control": control,
    "arms_total": len(v["arms"]),
    "arms_passed": sum(arm["passed"] for arm in v["arms"]),
}
a.results.mkdir(parents=True, exist_ok=False)
(a.results / "matrix.tsv").write_text("\n".join(lines) + "\n")
(a.results / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
print(f"{inputs['arms_passed']}/{inputs['arms_total']} arms passed; wrote {a.results}")
