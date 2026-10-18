#!/usr/bin/env python3
"""tier3-cells.py <bundle dir>: one line per cell of a verdicts-v1 bundle, in compare-pilot.py's own fields (the image
sha256/16, verdict, reason, reached, completed, cause, symbol+offset, attribution, controls), plus the bundle's arm and
platform `qemu` key. These lines are what a Tier 3 pre-registration commits as the prediction and what its result
commits as the reading: the raw bundle carries local paths. Exit 2 on no data."""
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("cp", Path(__file__).with_name("compare-pilot.py"))
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def main():
    b = Path(sys.argv[1])
    inp = b / "inputs.json"
    if not inp.is_file():
        print(f"NO DATA: {inp} missing"); return 2
    j = json.loads(inp.read_text())
    c = cp.cells(b)
    if not c:
        print(f"NO DATA: no cells in {b}"); return 2
    tally = {}
    for r in c.values():
        k = r["verdict"] + (f":{r['reason']}" if r["reason"] else "")
        tally[k] = tally.get(k, 0) + 1
    print(f"# arm {j.get('arm')} configuration {j.get('configuration')} platform {j.get('platform', {}).get('qemu')} "
          f"cells {len(c)} tally {json.dumps(tally, sort_keys=True)}")
    first = next(iter(c.values()))
    print(f"# controls {first['controls']}")
    for case in sorted(c):
        r = c[case]
        print(f"{case} image {str(r['image'])[:16]} {r['verdict']}{':' + r['reason'] if r['reason'] else ''} cause {r['cause']} "
              f"at {r['pc_off']} reached {r['reached']} completed {r['completed']} attribution {r['attribution']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
