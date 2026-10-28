#!/usr/bin/env python3
"""twin-reference.py STAGES RESULT_DIR... > reference.json: the result lines of run-stage.py runs, without the capture.

Per benchmark, size and arm: each command's rc, instret, cycles, CSRUNV entries, seconds, peak pinned pages and lifetime
nodes; the sha256 of every output the stage names (STAGES/<run dir name>/plan.json's `outputs`, the .err files left out: they carry the
launcher's counters); and each host validation's verdict. A directory without result.json is an error, not an empty row.
"""
import json
import sys
from pathlib import Path

out = []
stages = Path(sys.argv[1])
for d in sys.argv[2:]:
    p = Path(d) / "result.json"
    if not p.is_file():
        sys.exit(f"twin-reference: {p} does not exist")
    r = json.loads(p.read_text())
    names = set(json.loads((stages / Path(d).name / "plan.json").read_text())["outputs"])
    for arm, a in r["arms"].items():
        out.append(dict(
            bench=r["bench"], size=r["size"], arm=arm, run=Path(d).name,
            commands=[{k: c[k] for k in ("tag", "i", "rc", "instret", "cycles", "entries", "seconds", "peak_pages",
                                         "nodes")} for c in a["commands"]],
            outputs={n: h for n, h in sorted(a["hashes"].items()) if n.replace(arm, "ARM") in names
                     and not n.endswith(".err")},
            validation=[{"cmd": v["cmd"].split()[-1], "verdict": v["verdict"]} for v in a.get("validation", [])]))
if not out or not all(o["outputs"] for o in out):
    sys.exit("twin-reference: no rows, or a row with no output hashes")
if False:
    sys.exit("twin-reference: no result directories given")
json.dump(out, sys.stdout, indent=1)
print()
