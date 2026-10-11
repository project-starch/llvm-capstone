#!/usr/bin/env python3
"""Turn a domain run of this corpus into the committed result record.

Reads the runner's verdicts.json and the per-arm manifests, and writes
matrix.tsv (one line per arm) and inputs.json (the hashes that identify the
run: images, loader, emulator, compiler) into a fresh results/<stamp>/
directory beside a hand-written README. Raw serial logs stay outside the
repository.
"""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path, help="the runner's output directory")
p.add_argument("results", type=Path, help="results/<stamp> directory to create")
a = p.parse_args()
verdicts = json.loads((a.run / "verdicts.json").read_text())
columns = "case fix name shape mode expected passed cause pc expected_pc delivered completed runner_exit".split()
lines = ["\t".join(columns)]
inputs = {"arms": {}}
for r in sorted(verdicts, key=lambda r: (r["case"], r["mode"])):
    lines.append("\t".join(str(r.get(c, "")) if r.get(c) is not None else "" for c in columns))
    manifest = json.loads((Path(r["run"]) / "manifest.json").read_text())
    inputs["arms"][f"{r['case']:02d}-{r['mode']}"] = manifest["sha256"]
    inputs.setdefault("qemu_sha256", manifest["qemu_sha256"])
    inputs.setdefault("compiler_sha256", manifest["compiler_sha256"])
inputs["arms_total"] = len(verdicts)
inputs["arms_passed"] = sum(r["passed"] for r in verdicts)
a.results.mkdir(parents=True, exist_ok=False)
(a.results / "matrix.tsv").write_text("\n".join(lines) + "\n")
(a.results / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
print(f"{inputs['arms_passed']}/{inputs['arms_total']} arms passed; wrote {a.results}")
