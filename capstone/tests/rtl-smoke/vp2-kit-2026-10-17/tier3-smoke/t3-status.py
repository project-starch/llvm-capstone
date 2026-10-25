#!/usr/bin/env python3
"""t3-status.py: the Tier 3 campaign's state, one line per arm, read only from records/ (the committed prediction and
board files): the prediction's tally, then the board's comparison line, or "not run". Every board record of an arm
prints, the base <arm>.board.txt and each <arm>-<suffix>.board.txt (a re-run on another module, a dead boot), with
the module its kit names, so a re-run never hides behind the first reading. Records of arms outside ORDER print
last. Exit 2 when records/ is empty."""
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
    def board_line(bd):
        t = bd.read_text()
        cmp_line = next((l.strip() for l in t.splitlines()[::-1]
                         if "cells the same" in l or "CELLS DIFFER" in l or "NO DATA" in l), "no comparison")
        barm, btally = head(bd, "arm")
        mod = re.search(r"module ([0-9a-f]{16})", t.splitlines()[0] if t else "")
        kit = re.search(r"\b(bkit[\w-]*)", t.splitlines()[0] if t else "")
        ident = f"module {mod.group(1)[:8]}" if mod else f"kit {kit.group(1)}" if kit else "kit ?"
        line = f"{barm} {json.dumps(btally, sort_keys=True)} :: {cmp_line[:60]} :: {ident}"
        return line + (" :: WEDGE" if "wedge apertures" in t else "")

    seen = set()
    for tag in ORDER:
        pr = R / f"{tag}.prediction.txt"
        parm, ptally = head(pr, "arm") if pr.is_file() else (None, {})
        pred = json.dumps(ptally, sort_keys=True) if pr.is_file() else "no prediction"
        boards = ([R / f"{tag}.board.txt"] if (R / f"{tag}.board.txt").is_file() else []) + \
            sorted(R.glob(f"{tag}-*.board.txt"))
        seen.update(boards)
        if not boards:
            print(f"{tag:5} predicted {pred:55} board not run")
        for i, bd in enumerate(boards):
            name = bd.name[: -len(".board.txt")]
            lead = f"{tag:5} predicted {pred:55}" if i == 0 else f"{'':5} {'':65}"
            print(f"{lead} {name:18} {board_line(bd)}")
    rest = sorted(set(R.glob("*.board.txt")) - seen)
    for bd in rest:
        print(f"{'(other)':71} {bd.name[: -len('.board.txt')]:18} {board_line(bd)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
