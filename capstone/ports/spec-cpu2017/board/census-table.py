#!/usr/bin/env python3
"""census-table.py RESULT_DIR... : one row per benchmark, size and arm from run-stage.py's result.json files.

Columns: commands (ended/expected rc 0), instructions inside the context (the bench line's instret, summed over the
benchmark's own commands; validation and input generation are listed apart), peak pinned MiB (the launcher's peak
pages, max over commands), lifetime nodes (max), and the board time this predicts at 25 MHz for CPI 2.7 and 3.75 --
2.7 the geomean CPI of BEEBS/RV8/CoreMark on bitstream 3, 3.75 speedtest1 at size 100 -- which brackets, and does not
measure, a SPEC program's CPI. A twin run under -icount counts instructions exactly; its cycle CSR counts them too, so
cycles here are not a speed.
"""
import json
import sys
from pathlib import Path

MHZ = 25e6
print(f"{'benchmark':16s} {'size':5s} {'arm':19s} {'ok':>5s} {'instr (1e9)':>11s} {'val+gen':>8s} {'peak MiB':>8s} "
      f"{'nodes':>6s} {'board h @2.7':>12s} {'@3.75':>6s}")
for d in sys.argv[1:]:
    p = Path(d) / "result.json"
    if not p.exists():
        print(f"{Path(d).name:40s} NO result.json")
        continue
    r = json.loads(p.read_text())
    for arm, a in r["arms"].items():
        run = [c for c in a["commands"] if c["tag"] == "run"]
        other = [c for c in a["commands"] if c["tag"] != "run"]
        ins = [c["instret"] for c in run]
        tot = sum(i for i in ins if i) if all(ins) else None
        oth = sum(c["instret"] or 0 for c in other)
        peak = max((c["peak_pages"] or 0) for c in a["commands"]) * 4096 / 2**20 if a["commands"] else 0
        nodes = max((c["nodes"] or 0) for c in a["commands"]) if a["commands"] else 0
        ok = sum(c["rc"] == 0 for c in a["commands"])
        h = (lambda cpi: f"{(tot + oth) * cpi / MHZ / 3600:.2f}") if tot is not None else (lambda cpi: "?")
        print(f"{r['bench']:16s} {r['size']:5s} {arm:19s} {ok:>2d}/{len(a['commands']):<2d} "
              f"{(tot or 0) / 1e9:11.3f} {oth / 1e9:8.3f} {peak:8.0f} {nodes:6d} {h(2.7):>12s} {h(3.75):>6s}")
