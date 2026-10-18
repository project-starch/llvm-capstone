#!/usr/bin/env python3
"""subset-bundle.py <prediction bundle> <board bundle> <out dir>: the prediction bundle cut to the board bundle's cases,
for an arm the board runs in several --only boots (the payload cap). Writes <out>/verdicts.jsonl (the prediction's lines
for exactly the board's cases, unchanged) and copies inputs.json. Refuses (exit 1) when a board case is absent from the
prediction; 2 on no data. compare-pilot.py then compares the subset cell by cell, by image, as for a whole arm."""
import json
import shutil
import sys
from pathlib import Path


def cases(bundle):
    f = Path(bundle) / "verdicts.jsonl"
    if not f.is_file():
        sys.exit(f"NO DATA: {f} missing")
    return [(json.loads(l)["observation"]["case"], l) for l in f.read_text().splitlines() if l.strip()]


def main():
    pred, board, out = (Path(x) for x in sys.argv[1:4])
    p, b = cases(pred), {c for c, _ in cases(board)}
    if not b:
        print("NO DATA: the board bundle has no cells"); return 2
    missing = b - {c for c, _ in p}
    if missing:
        print(f"REFUSED: board cases absent from the prediction: {sorted(missing)}"); return 1
    out.mkdir(parents=True, exist_ok=False)
    (out / "verdicts.jsonl").write_text("".join(l + "\n" for c, l in p if c in b))
    shutil.copy2(pred / "inputs.json", out / "inputs.json")
    print(f"SUBSET {len(b)} of {len(p)} prediction cells -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
