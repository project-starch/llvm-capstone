#!/usr/bin/env python3
"""juliet-obs-compare.py <reference observations.json> <board observations.json> <build dir> <llvm bin> [--tolerate CASE ...]

Juliet, cell by cell, in what a board can observe: for every bad and good image, the exit kind and value, the domain
fault's cause and its site (symbol+offset in the image), and whether the step timed out. judge.py cannot do this for a
board: its CAUGHT needs QEMU's own "Cap mem access OOB/revoked" line (judge.py STRICT), which silicon never prints, so
on a board every catch reads NO-READING:unattributed. The reference is a QEMU run of the SAME build (its observations).
Control steps (ok-*, ctl-*) are compared too. A case named with --tolerate (a value-dependent case) is reported, not
counted as a difference. Exit 0 when every other image matches, 1 when any differs, 2 on no data."""
import json
import re
import sys
from pathlib import Path


def load(path, sym):
    try:
        o = json.load(open(path))
    except FileNotFoundError:
        print(f"NO DATA: {path} does not exist"); sys.exit(2)
    cells = {}
    for s in o["steps"]:
        f = s.get("fault_line")
        site = cause = None
        if f:
            m = re.search(r"cause=(\d+) pc=0x([0-9a-f]+).*?code=0x([0-9a-f]+)-", f)
            if m:
                cause = int(m.group(1))
                fn, off = sym(s["image"], int(m.group(2), 16), int(m.group(3), 16))
                site = f"{fn}+{off:#x}" if fn else f"code+{int(m.group(2), 16) - int(m.group(3), 16):#x}"
        key = s["image"] if s["image"] not in ("ok", "controls") else f"{s['image']}:{re.sub(r'-[0-9]+$', '', s['step'])}"
        cells.setdefault(key, []).append((tuple(s["exit"]) if s.get("exit") else None, cause, site, bool(s.get("timeout"))))
    return cells


def main():
    if len(sys.argv) < 5:
        print(__doc__); sys.exit(2)
    ref_p, board_p, build, llvm = sys.argv[1:5]
    tol = set(sys.argv[sys.argv.index("--tolerate") + 1:]) if "--tolerate" in sys.argv else set()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "bug-corpora" / "tools"))
    import verdicts  # noqa: E402
    cache = {}

    def sym(image, pc, code):
        p = Path(build) / "img" / f"{image}.dom"
        if not p.is_file():
            return None, None
        if p not in cache:
            cache[p] = verdicts.Symbols(llvm, p)
        return cache[p].lookup(pc, code)

    ref, board = load(ref_p, sym), load(board_p, sym)
    if not ref or not board:
        print("NO DATA: an observations file has no steps"); sys.exit(2)
    keys = sorted(set(ref) | set(board))
    same = diff = tolerated = 0
    for k in keys:
        a, b = ref.get(k), board.get(k)
        case = k.rsplit(".", 1)[0]
        if a == b:
            same += 1
            continue
        if case in tol:
            tolerated += 1
            print(f"TOLERATED {k}: reference {a} board {b}")
            continue
        diff += 1
        print(f"DIFF {k}: reference {a} board {b}")
    print(f"{same} of {len(keys)} cells the same, {diff} different, {tolerated} tolerated")
    sys.exit(0 if diff == 0 else 1)


if __name__ == "__main__":
    main()
