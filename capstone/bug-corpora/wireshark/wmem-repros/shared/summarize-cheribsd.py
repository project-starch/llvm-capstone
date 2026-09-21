#!/usr/bin/env python3
"""Turn CheriBSD corpus runs into the committed result record.

Takes the output directories of host/cheribsd/poisoncap/run.py -- the plain
arm (one mode) and the PoisonCap arms (two modes) -- and writes matrix.tsv
(one line per arm) and inputs.json (binary and platform hashes) beside a
hand-written README. Raw guest logs stay outside the repository.
"""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("results", type=Path, help="results/<stamp> directory to create")
p.add_argument("--plain", type=Path, help="run.py output of the plain build, modes 0")
p.add_argument("--poisoncap", type=Path, help="run.py output of the PoisonCap build, modes 0,1")
p.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent.parent)
a = p.parse_args()
if not (a.plain or a.poisoncap):
    p.error("give --plain and/or --poisoncap")
cases = {}
for d in sorted(a.corpus.glob("[0-9][0-9]_*")):
    c = json.loads((d / "case.json").read_text())
    cases[c["case"]] = c
columns = "case fix name shape arm expected passed exit fault_pc probe_address at_probe counters".split()
lines = ["\t".join(columns)]
inputs = {}


def guest_rows(run, arm_of_mode, label):
    summary = json.loads((run / "guest/summary.json").read_text())
    inputs[label] = {"platform": summary["platform"], "runtime_revocation": summary["runtime_revocation"],
                        "guest_default_revocation": summary["guest_default_revocation"],
                        "binaries": summary["binaries"], "status": summary["status"]}
    matrix = json.loads((run / "matrix.json").read_text())
    if not matrix.get("complete"):
        raise SystemExit(f"{run}: incomplete suite, no record")
    by_case = {r["case"]: r for r in matrix["rows"]}
    for row in summary["results"]:
        name = row["name"]
        if "-mode" not in name:
            continue
        stem, mode = name.rsplit("-mode", 1)
        which = int(stem[:2])
        c = cases[which]
        v = by_case.get(which, {})
        arm = arm_of_mode[int(mode)]
        expected = "fault" if arm == "poisoncap-protected" else "complete"
        if "paired" in v:
            fault_pc = ",".join(v["protected_fault_pcs"]) if mode == "1" else ""
            probe = v.get("probe_address") if mode == "1" else ""
            at_probe = v.get("fault_at_probe") if mode == "1" else ""
        else:
            fault_pc = ",".join(v.get("fault_pcs", []))
            probe = ""
            at_probe = v.get("fault_at_probe")
        # Each arm's own counters, as the driver printed them at the ready
        # marker: the pair verdict carries only the protected arm's line.
        counters = next((l for l in (run / "guest" / name / "stdout.txt").read_text(errors="replace").splitlines()
                         if l.startswith("WM_POISONCAP mode=")), "")
        lines.append("\t".join(str(x) if x is not None else "" for x in (
            which, c["upstream_fix"], stem, c["shape"], arm, expected, row["passed"], row["exit"],
            fault_pc, probe, at_probe, counters)))


if a.plain:
    guest_rows(a.plain, {0: "cheribsd"}, "plain")
if a.poisoncap:
    guest_rows(a.poisoncap, {0: "poisoncap-spatial", 1: "poisoncap-protected"}, "poisoncap")
a.results.mkdir(parents=True, exist_ok=False)
(a.results / "matrix.tsv").write_text("\n".join(lines) + "\n")
(a.results / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
print(f"{len(lines) - 1} arms recorded in {a.results}")
