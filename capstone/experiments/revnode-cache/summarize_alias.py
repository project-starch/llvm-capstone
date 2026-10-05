#!/usr/bin/env python3
"""Tabulate aliasstat reports side by side.

    summarize_alias.py name=report.alias.json [...]

Percentiles come from the histogram buckets, so above 8 they are bucket upper bounds
("<=16" means the value lies in 9-16). A report with errors is printed but makes the
exit status non-zero.
"""
import json
import sys


def pctl(h, q):
    if not h["count"]:
        return "-"
    want, acc = q * h["count"], 0
    for label, n in h["buckets"].items():
        acc += n
        if acc >= want:
            return label if "-" not in label else "<=" + label.split("-")[1]
    return str(h["max"])


def row(name, h):
    return (f"  {name:<14} n={h['count']:>10,}  mean {h['mean']:8.2f}  p50 {pctl(h, .5):>7}"
            f"  p90 {pctl(h, .9):>7}  p99 {pctl(h, .99):>7}  max {h['max']:>8,}")


def main():
    rc = 0
    reps = [(a.split("=", 1)[0], json.load(open(a.split("=", 1)[1]))) for a in sys.argv[1:]]
    for name, r in reps:
        e = r["errors"]
        bad = e["negative_alias_count"] or e["walk_mismatch"]
        rc |= bool(bad)
        m = r["memory_alias_events"]
        print(f"== {name}: nodes {r['nodes_allocated']:,} (mrev {r['mrev']:,}, split {r['split']:,})"
              f"  copies stored {m['stored']:,}  peak live copies {m['peak_live']:,}"
              f"  errors {e}{'  <-- INSTRUMENT ERROR' if bad else ''}")
    sections = [
        ("aliases", "max_memory_aliases_per_node", "most memory copies a node ever had at once"),
        ("aliases", "at_revoke_root_memory", "copies of the revoked capability itself, memory"),
        ("aliases", "at_revoke_root_registers", "                                    registers"),
        ("aliases", "at_revoke_invalidated_memory", "copies made stale by one revoke, memory"),
        ("aliases", "at_revoke_invalidated_registers", "                                registers"),
        ("aliases", "at_free_memory", "copies left when the collector frees a node (must be 0)"),
        ("trees", "revoke_run", "nodes one revoke invalidates"),
        ("trees", "revoke_root_children", "direct children of a revoked node"),
        ("trees", "live_node_children", "direct children of a live node (snapshots)"),
        ("trees", "live_node_depth", "depth of a live node (snapshots)"),
        ("trees", "tree_size", "nodes per tree (snapshots)"),
    ]
    for group, key, title in sections:
        print(f"\n{title}  [{key}]")
        for name, r in reps:
            print(row(name, r[group][key]))
    return rc


if __name__ == "__main__":
    sys.exit(main())
