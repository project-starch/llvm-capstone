#!/usr/bin/env python3
"""The proposed index against Capstone and Clover's baseline, per workload.

    summarize_bucket.py name=<reports prefix> [...]

<prefix>.clover.json supplies Capstone and the Clover baseline, <prefix>.bucket.json the
proposed index's variants. Everything is per 1000 lifetime checks (capability loads and
stores), the denominator all designs share. A missing report is an error.
"""
import json
import sys

SIZES = [64, 1024, 16384]


def load(prefix):
    c = json.load(open(prefix + ".clover.json"))
    b = json.load(open(prefix + ".bucket.json"))
    if c["lifetime_checks"] != b["lifetime_checks"]:
        sys.exit(f"{prefix}: the two reports count different checks")
    return c, b


def misses(design, sizes_lines, k):
    return [design["misses"][sizes_lines.index(s)] / k for s in SIZES]


def main():
    reps = [(a.split("=", 1)[0], *load(a.split("=", 1)[1])) for a in sys.argv[1:]]
    if not reps:
        sys.exit("usage: summarize_bucket.py name=prefix ...")
    variants = [f"K{v['inline']}/W{v['slot_cache']}/{v['node_bytes']}B" for v in reps[0][2]["variants"]]
    print("metadata accesses and misses per 1000 lifetime checks (64-byte lines, fully associative LRU)")
    print(f"{'':16}{'design':18}{'accesses':>10}" + "".join(f"{str(s) + ' lines':>12}" for s in SIZES))
    for name, c, b in reps:
        k = c["lifetime_checks"] / 1000
        rows = [("capstone", c["capstone"]), ("clover baseline", c["clover"])]
        rows += [(variants[i], v) for i, v in enumerate(b["variants"])]
        for i, (label, d) in enumerate(rows):
            sl = c["sizes_lines"] if i < 2 else b["sizes_lines"]
            print(f"{name if i == 0 else '':16}{label:18}{d['accesses'] / k:10.2f}"
                  + "".join(f"{m:12.3f}" for m in misses(d, sl, k)))
        print()
    print("the proposed index: what the slot cache absorbs, what overflows, what a revoke costs")
    print(f"{'':16}{'variant':18}{'changes/1000':>13}{'absorbed':>10}{'del.abs.':>10}{'ovfl ins':>10}"
          f"{'ovfl nodes':>11}{'grows':>7}{'rev.acc mean':>13}{'max':>8}{'cam skips':>10}{'peak entries':>13}{'err':>5}")
    for name, c, b in reps:
        k = c["lifetime_checks"] / 1000
        for i, v in enumerate(b["variants"]):
            ch = v["index_changes"]
            ov = v["overflow"]
            rv = v["revokes"]
            print(f"{name if i == 0 else '':16}{variants[i]:18}{ch / k:13.2f}"
                  f"{100 * v['absorbed_by_slot_cache'] / max(ch, 1):9.2f}%"
                  f"{100 * v['deletes_absorbed'] / max(v['data_stores_over_capability'], 1):9.1f}%"
                  f"{100 * ov['inserts'] / max(ch, 1):9.2f}%{ov['nodes']:>11,}{ov['grows']:>7,}"
                  f"{rv['accesses_mean']:13.1f}{rv['accesses_max']:>8,}{rv['cam_skips']:>10,}"
                  f"{v['memory_entries_peak']:>13,}{v['errors']:>5}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
