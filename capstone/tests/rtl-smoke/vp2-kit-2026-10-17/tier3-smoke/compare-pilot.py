#!/usr/bin/env python3
"""compare-pilot.py QEMU_BUNDLE BOARD_BUNDLE: the Tier 3 pilot's cell-by-cell comparison of two verdicts-v1 bundles of
one corpus arm, a QEMU arm (the prediction) and a board arm.

Per cell (observation.case), it compares:
- the image sha256: the same image on both, or the comparison is void;
- verdict and reason;
- reached and completed;
- the fault's cause;
- the fault's symbol+offset (or its pc relative to the code base);
- the attribution.
Then it compares the controls by name and observed value.
Prints one line per cell, SAME or DIFF with the differing fields (VOID when the images differ, NOT RUN when a
side has no image).
Exit 0 when every cell and control is the same, 1 when any differs, 2 when a bundle is missing or the cells do not
match (no data is never a pass)."""
import json
import re
import sys
from pathlib import Path

FAULT = re.compile(r"cause=(\d+) pc=(0x[0-9a-f]+) .*?code=(0x[0-9a-f]+)-")


def cells(bundle):
    f = Path(bundle) / "verdicts.jsonl"
    if not f.is_file():
        sys.exit(f"NO DATA: {f} missing")
    out = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        o = r["observation"]
        fault = o.get("fault")
        if isinstance(fault, dict):  # verdicts.py's resolved fault: cause, pc, symbol, offset
            cause, where = fault.get("cause"), f"{fault.get('symbol')}+{fault.get('offset')}"
        else:
            m = FAULT.search(fault or "")
            cause = int(m.group(1)) if m else None
            where = hex(int(m.group(2), 16) - int(m.group(3), 16)) if m else None
        out[o["case"]] = {
            "image": o.get("image_sha256"), "verdict": r.get("verdict"), "reason": r.get("reason"),
            "reached": o.get("reached"), "completed": o.get("completed"),
            "cause": cause, "pc_off": where,
            "attribution": json.dumps(o.get("attribution"), sort_keys=True),
            "controls": json.dumps(sorted((c.get("name"), c.get("observed")) for c in o.get("controls") or [])),
        }
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    q, b = cells(sys.argv[1]), cells(sys.argv[2])
    if not q or set(q) != set(b):
        print(f"CELLS DIFFER: qemu {sorted(q)} board {sorted(b)}")
        return 2
    bad = 0
    for case in sorted(q):
        if not q[case]["image"] or not b[case]["image"]:
            # a cell with no image on one side never ran there (a boot cut short, a setup that stopped first)
            side = "board" if not b[case]["image"] else "qemu"
            print(f"NOT RUN {case}: no image on the {side} side ({(b if side == 'board' else q)[case]['verdict']}, "
                  f"{(b if side == 'board' else q)[case]['reason']})")
            bad += 1
            continue
        if q[case]["image"] != b[case]["image"]:
            print(f"VOID {case}: images differ ({q[case]['image'][:16]} vs {b[case]['image'][:16]})")
            bad += 1
            continue
        diffs = [k for k in q[case] if k != "image" and q[case][k] != b[case][k]]
        bad += bool(diffs)
        print(f"{'DIFF' if diffs else 'SAME'} {case}: verdict {b[case]['verdict']} cause {b[case]['cause']} "
              f"pc {b[case]['pc_off']}" + "".join(f"\n    {k}: qemu {q[case][k]} board {b[case][k]}" for k in diffs))
    print(f"{len(q) - bad} of {len(q)} cells the same")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
