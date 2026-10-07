#!/usr/bin/env python3
"""What the objects a program holds at its peak would need in Capstone's Sublet heap.

Reads the MQ-DONE line of tracer v4 (peak_live, peak_live_objects, peak_live_pow2_256: the peak of
the bytes held with every request rounded up to a power of two of at least 256 bytes, the buddy
rounding of ports/musl-capstone/runtime/sublet_heap.c) and compares it with the pool
(CAPSTONE_SUBLET_HEAP_LOG 22 = 4 MiB) and the identities (65,536 per process).
  usage: fit.py <result files...>
"""
import re
import sys

POOL = 4 << 20
IDS = 65536
MiB = 1 << 20

print(f"{'program':28s} {'peak MiB':>9s} {'objects':>8s} {'pow2-256 MiB':>13s} {'x held':>7s}"
      f" {'pool 4 MiB':>11s} {'ids':>6s}")
for path in sys.argv[1:]:
    text = open(path, errors="replace").read()
    m = re.search(r"MQ-DONE .*peak_live=(\d+) peak_live_objects=(\d+) peak_live_pow2_256=(\d+)", text)
    if not m:
        continue
    live, objs, p2 = map(int, m.groups())
    name = re.sub(r"^.*_\._traced4_\._", "", path.rsplit("/", 1)[-1])[:-4].replace("_", " ")[:28]
    print(f"{name:28s} {live / MiB:9.2f} {objs:8d} {p2 / MiB:13.2f} {p2 / live:7.2f}"
          f" {'fits' if p2 <= POOL else 'too small':>11s} {'fits' if objs <= IDS else 'no':>6s}")
