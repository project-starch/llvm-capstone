#!/usr/bin/env python3
"""juliet-sub-read.py <run out>: every step of a juliet/run.py observations.json, one line each, plus each batch's
completion, sequence check and wall time, and the board readings a vbatch boot carries (pc=0, DRT1, FRTY, INJS, INJL,
read from the batch's own transcript). Exit 2 when there is no observations.json (no data is an error)."""
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
p = out / "observations.json"
if not p.is_file():
    print(f"NO DATA: {p} is missing"); sys.exit(2)
o = json.loads(p.read_text())
for b in o["batches"]:
    seq_ok = b["plan"] == b["serial_sequence"][:-1] and b["serial_sequence"][-1:] == ["stats"]
    t = (out / b["batch"] / "work" / "board.txt")
    txt = t.read_text(errors="replace") if t.is_file() else ""
    # the pass runners' own patterns (tier3-pass-rep.sh), so a Juliet boot reads like every other vbatch boot
    pats = {"pc=0": r"pc=0x0\b|pc=0 ", "DRT1": r"^DRT1:", "FRTY": r"FRTY:", "Oops": r"Oops", "INJS": r"INJS",
            "INJL": r"INJL", "node-floor-2": r"node-floor-2"}
    lines = txt.replace("\r", "").split("\n")
    marks = {k: sum(1 for l in lines if re.search(v, l)) for k, v in pats.items()}
    hb = len(re.findall(r"^@@HB \d+", txt.replace("\r", ""), re.M))
    print(f"BATCH {b['batch']} completed={b['completed']} sequence-as-planned={seq_ok} wall_s={b['wall_s']} "
          f"heartbeats(30s)={hb} stats={'yes' if b['stats'] else 'NO'} readings={marks}")
for s in o["steps"]:
    cause = re.search(r"cause=(\d+)", s["fault_line"] or "")
    print(f"  {s['step']:13} {s['image'][:52]:52} exit={s['exit']} cause={cause.group(1) if cause else '-'} "
          f"timeout={s['timeout']} events={','.join(e.split()[0] + ('' if len(e.split()) < 2 else e.split()[1][:4]) for e in s['events'])}")
print(f"STEPS {len(o['steps'])} limit_ms {o['limit_ms']} runner {o['runner_sha256'][:16]} helper {o['helper_sha256'][:16]}")
