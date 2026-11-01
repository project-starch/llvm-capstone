#!/usr/bin/env python3
"""plain-gate.py <census dir> <stages dir> <candidates>: the board list lines of the Plain stages whose QEMU twin run
passed (SPEC-DONE, every command and target validation ended rc 0, every host specdiff PASS, hashes present), each
with its twin; every other candidate is printed to stderr as HELD with the reason. No candidate is an error (exit 2)."""
import json, sys
from pathlib import Path
C, ST, cand = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = [l.split() for l in cand.read_text().splitlines() if l.strip() and not l.startswith("#")]
if not rows:
    sys.exit("plain-gate: no candidates")
for name, step, book in rows:
    r, plan = C / name / "result.json", json.loads((ST / name / "plan.json").read_text())
    why = None
    if not r.is_file():
        why = "no twin result"
    else:
        d = json.loads(r.read_text()); a = d["arms"].get("plain-rv64")
        n = len(plan["commands"]) + len(plan["target_validation"]) + len(plan["generation"])
        if not d.get("done"): why = "SPEC-DONE absent"
        elif not a: why = "no plain-rv64 arm"
        elif len(a["commands"]) != n: why = f"{len(a['commands'])}/{n} commands"
        elif any(c["rc"] != 0 for c in a["commands"]): why = "rc != 0"
        elif not a["hashes"]: why = "no hashes"
        elif not a.get("validation") or any(v["verdict"] != "PASS" for v in a["validation"]): why = "specdiff not all PASS"
    if why:
        print(f"HELD {name}: {why}", file=sys.stderr)
    else:
        print(f"{ST / name} {step} {book} {C / name}")
