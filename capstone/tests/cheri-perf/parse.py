#!/usr/bin/env python3
"""Parse RCPERF lines from the CHERI perf serial log and report the per-op
instruction count for each revocation config plus the overhead over the
spatial (revocation-off) baseline. Uses the median trial per config."""
import re
import sys
import statistics


def main():
    if len(sys.argv) < 2:
        print("usage: parse.py <serial.log>", file=sys.stderr)
        return 2
    txt = open(sys.argv[1], encoding="utf-8", errors="replace").read()

    counter = None
    m = re.search(r"RCPERF-COUNTER\s+(\S+)", txt)
    if m:
        counter = m.group(1)
    if "RCPERF-FATAL" in txt:
        print("FATAL: no user-readable instruction counter in the guest; see log")
        return 1

    per = {}  # mode -> list of perop
    iters = block = None
    for m in re.finditer(
        r"RCPERF mode=(\w+) iters=(\d+) block=(\d+) empty=(\d+) allocfree=(\d+) perop=([\d.]+)",
        txt,
    ):
        mode = m.group(1)
        iters, block = int(m.group(2)), int(m.group(3))
        per.setdefault(mode, []).append(float(m.group(6)))

    hdr = f"{'config':<10} {'per-op (instr)':>16} {'overhead vs spatial':>22}"

    if per:
        med = {k: statistics.median(v) for k, v in per.items()}
        print(f"counter        : {counter}")
        print(f"=== microbench: malloc/touch/free  (iters={iters}, block={block} B, "
              f"trials/config={max(len(v) for v in per.values())}) ===")
        base = med.get("spatial")
        print(hdr)
        print("-" * len(hdr))
        for mode in ("spatial", "temporal", "eager"):
            if mode not in med:
                continue
            v = med[mode]
            ov = (f"+{v - base:.1f}  ({v / base:.2f}x)"
                  if base is not None and mode != "spatial" else "(baseline)")
            print(f"{mode:<10} {v:>16.2f} {ov:>22}")
        for mode in ("spatial", "temporal", "eager"):
            if mode in per and len(per[mode]) > 1:
                print(f"  {mode} trials: {['%.1f' % x for x in per[mode]]}")
    else:
        print("(microbench skipped or produced no RCPERF lines)")

    # ---- real-workload (binary-search-tree) arm ----
    tree = {}
    tkeys = trounds = None
    for m in re.finditer(
        r"RCTREE mode=(\w+) keys=(\d+) rounds=(\d+) ops=(\d+) instr=(\d+) perop=([\d.]+)",
        txt,
    ):
        mode = m.group(1)
        tkeys, trounds = int(m.group(2)), int(m.group(3))
        tree.setdefault(mode, []).append(float(m.group(6)))
    if tree:
        tmed = {k: statistics.median(v) for k, v in tree.items()}
        tbase = tmed.get("spatial")
        print()
        print(f"=== real workload: BST build/lookup/destroy "
              f"(keys={tkeys}, rounds={trounds}) ===")
        print(hdr)
        print("-" * len(hdr))
        for mode in ("spatial", "temporal", "eager"):
            if mode not in tmed:
                continue
            v = tmed[mode]
            if tbase is not None and mode != "spatial":
                ov = f"+{v - tbase:.1f}  ({v / tbase:.2f}x)"
            else:
                ov = "(baseline)"
            n = f"  [n={len(tree[mode])}]" if len(tree[mode]) != 1 else ""
            print(f"{mode:<10} {v:>16.2f} {ov:>22}{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
