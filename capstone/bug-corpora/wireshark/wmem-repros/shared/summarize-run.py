#!/usr/bin/env python3
"""Turn one run-defects.py output tree into the committed result record.

Writes matrix.tsv (one line per arm), inputs.json (the hashes that identify
the binaries and tools behind every line) and nothing else: raw serial logs
stay outside the repository. The README beside them is written by hand.
"""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path, help="output directory of run-defects.py")
p.add_argument("results", type=Path, help="results/<stamp> directory to create")
p.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent.parent)
a = p.parse_args()
verdicts = json.loads((a.run / "verdicts.json").read_text())
cases = {}
for d in sorted(a.corpus.glob("[0-9][0-9]_*")):
    c = json.loads((d / "case.json").read_text())
    cases[c["case"]] = c
a.results.mkdir(parents=True, exist_ok=False)
columns = "case fix name shape mode expected passed cause pc expected_pc site delivered completed runner_exit".split()
lines = ["\t".join(columns)]
images, tools = {}, {}
for r in sorted(verdicts, key=lambda r: (r["case"], r["mode"])):
    c = cases[r["case"]]
    m = json.loads((Path(r["run"]) / "manifest.json").read_text())
    images[r["name"]] = m["sha256"]["defect.dom"]
    tools = {"domain_loader": m["sha256"]["host.user"], "qemu": m["qemu_sha256"],
             "clang": m["compiler_sha256"], "node_capacity": m["node_capacity"]}
    row = {**r, "fix": c["upstream_fix"], "shape": c["shape"],
           "completed": "" if r["mode"] == "sublet" else r["passed"]}
    lines.append("\t".join("" if row.get(k) is None else str(row.get(k)) for k in columns))
(a.results / "matrix.tsv").write_text("\n".join(lines) + "\n")
(a.results / "inputs.json").write_text(json.dumps({
    "images": images, **tools,
    "distinct_trace_inputs": len({json.loads((Path(r["run"]) / "manifest.json").read_text())["sha256"]["trace.bin"] for r in verdicts}),
    "arms": len(verdicts),
}, indent=2) + "\n")
print(f"{sum(r['passed'] for r in verdicts)}/{len(verdicts)} arms passed; wrote {a.results}")
