#!/usr/bin/env python3
"""compare-wedge-apertures.py <board.txt>...: the vm-suite wedge apertures of several boots side by side (the driver's
"===== vm-suite wedge apertures =====" block), the first boot as the reference; prints how many switch apertures equal it.
Exit 0 when every boot equals the first in every switch aperture, 1 otherwise, 2 when a boot has no aperture block."""
import re
import sys
from pathlib import Path

EXTRA = ("commit pc", "trap-log mepc", "trap-log tval", "rev_node_debug_ex", "rev_node_head_ex", "rev_node_serving_idx_ex")


def apertures(path):
    t = Path(path).read_text(errors="replace").replace("\r", "")
    if "===== vm-suite wedge apertures =====" not in t:
        return None
    blk = t[t.index("===== vm-suite wedge apertures ====="):][:6000].split("\n")[1:]
    sw = {m.group(1): m.group(2) for line in blk if (m := re.match(r"sw=(\d+) .* (0x[0-9a-f]+) [01]{8}$", line.strip()))}
    ext = {k: m.group(1) for line in blk for k in EXTRA
           if line.strip().startswith(k) and (m := re.search(r"(0x[0-9a-f]+)\s*$", line.strip()))}
    return sw, ext


def main():
    data = [apertures(p) for p in sys.argv[1:]]
    if any(d is None for d in data):
        print("NO DATA: a boot has no wedge aperture block"); return 2
    keys = sorted(set().union(*[d[0] for d in data]), key=int)
    bad = 0
    for k in keys:
        vals = [d[0].get(k, "-") for d in data]
        bad += any(v != vals[0] for v in vals[1:])
        print(f"{k:>5} " + " ".join(f"{v:>6}" for v in vals) + ("" if all(v == vals[0] for v in vals) else "   <- differs"))
    for k in EXTRA:
        print(f"{k:>24}: " + "  ".join(d[1].get(k, "-") for d in data))
    print(f"{len(keys)} switch apertures; {len(keys) - bad} equal in every boot, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
