#!/usr/bin/env python3
"""t3-status.py: the Tier 3 campaign's state, one line per arm, read only from records/ (the committed prediction and
board files): the prediction's tally, then the board's comparison line, or "not run". Exit 2 when records/ is empty."""
import json
import re
import sys
from pathlib import Path

R = Path(__file__).with_name("records")
ORDER = ["mpt", "mph", "fph", "fpt", "fca", "fcs", "wph", "wpt", "fso", "pb12", "pb13", "pb9", "pb10", "pgc", "pgm0",
         "pgm1", "wm0", "wm1", "pym0", "pym1", "sq0", "sq1", "sq1a", "sq0a", "sq0b", "sq0c", "sq0d", "sq0e", "sq0f",
         "ppA", "ppB", "ppC", "ppD"]


def head(p, key):
    for line in p.read_text().splitlines():
        if line.startswith("# arm "):
            m = re.search(r"tally (\{.*\})", line)
            arm = line.split()[2]
            return arm, (json.loads(m.group(1)) if m else {})
    return None, {}


def main():
    if not R.is_dir() or not any(R.iterdir()):
        print(f"NO DATA: {R} is empty"); return 2
    for tag in ORDER:
        pr, bd = R / f"{tag}.prediction.txt", R / f"{tag}.board.txt"
        parm, ptally = head(pr, "arm") if pr.is_file() else (None, {})
        if bd.is_file():
            t = bd.read_text()
            cmp_line = next((l.strip() for l in t.splitlines()[::-1]
                             if "cells the same" in l or "CELLS DIFFER" in l or "NO DATA" in l), "no comparison")
            barm, btally = head(bd, "arm")
            board = f"{barm} {json.dumps(btally, sort_keys=True)} :: {cmp_line[:60]}"
            if "wedge apertures" in t:
                board += " :: WEDGE"
        else:
            board = "not run"
        pred = json.dumps(ptally, sort_keys=True) if pr.is_file() else "no prediction"
        print(f"{tag:5} predicted {pred:55} board {board}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
