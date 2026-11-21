"""hmcheck.py <log> [--hwm]: read the @@HM lines of a CAPSTONE_HEAP_MAPPING_REUSE_TEST_LOG image.

Line: @@HM <op> <bytes> <live> <cached> <peak>, values after the event. op: M fresh map, U reuse hit, C cached on
release, F released by munmap, E/R/H/P cached mapping given back (evict-all, make-room, high-water, mprotect).
Recomputes live and cached from the ops alone and checks them against the printed values (the instrument checks
itself), then prints the peak of live + cached against the peak of live. With --hwm, exits 1 if live + cached ever
exceeded the printed peak. Exits 2 on no lines (no data is an error)."""
import re
import sys

ev = [(m.group(1), *map(int, m.groups()[1:]))
      for m in re.finditer(r"^@@HM ([MUCFERHP]) (\d+) (\d+) (\d+) (\d+)$", open(sys.argv[1], errors="replace").read(), re.M)]
if not ev:
    print("NO DATA: no @@HM lines in", sys.argv[1])
    sys.exit(2)
live = cached = peak = 0
foot_max = foot_at = 0
bad = over = 0
counts = {}
for i, (op, b, l, c, p) in enumerate(ev):
    counts[op] = counts.get(op, 0) + 1
    if op == "M": live += b
    elif op == "U": live += b; cached -= b
    elif op == "C": live -= b; cached += b
    elif op == "F": live -= b
    else: cached -= b
    peak = max(peak, live)
    if (live, cached, peak) != (l, c, p):
        bad += 1
        if bad <= 3: print(f"MISMATCH at {i}: {op} {b}: model {live} {cached} {peak}, printed {l} {c} {p}")
    if l + c > foot_max: foot_max, foot_at = l + c, i
    if l + c > p: over += 1
print(f"events {len(ev)} {counts}")
print(f"peak live {peak:,} B; peak live+cached {foot_max:,} B (event {foot_at}); excess {foot_max - peak:,} B; "
      f"events above the live peak {over}; model mismatches {bad}")
print(f"last: live {ev[-1][2]:,} cached {ev[-1][3]:,}")
sys.exit(1 if bad or ("--hwm" in sys.argv and over) else 0)
