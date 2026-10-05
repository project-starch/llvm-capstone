#!/usr/bin/env python3
"""Capstone vs Clover metadata traffic, from clovsim reports.

    summarize_clover.py name=report.clover.json [...]

All figures are per 1000 lifetime checks (capability loads/stores), the denominator both
designs share. Capstone's traffic sits on the load/store path; Clover's on capability
stores, data stores over capabilities, and revokes.
"""
import json
import sys


def main():
    reps = [(a.split("=", 1)[0], json.load(open(a.split("=", 1)[1]))) for a in sys.argv[1:]]
    sizes = reps[0][1]["sizes_lines"]
    print("metadata line accesses and misses per 1000 lifetime checks (64-byte lines, fully associative LRU)")
    hdr = "".join(f"{str(s) + ' lines':>13}" for s in sizes)
    print(f"{'':16}{'design':9}{'accesses':>10}{hdr}")
    for name, r in reps:
        k = r["lifetime_checks"] / 1000
        for d in ("capstone", "clover"):
            x = r[d]
            print(f"{name:16}{d:9}{x['accesses'] / k:10.2f}" + "".join(f"{m / k:13.3f}" for m in x["misses"]))
    print("\nclover / capstone misses")
    for name, r in reps:
        row = ""
        for a, b in zip(r["clover"]["misses"], r["capstone"]["misses"]):
            row += f"{(a / b if b else float('inf')):13.2f}"
        print(f"{name:25}{'':10}{row}")
    print("\nClover detail")
    for name, r in reps:
        c = r["clover"]
        st = c["capability_stores"]
        tot = st["index_unchanged_same_node"] + st["index_updated"] + st["of_a_revoked_capability"]
        k = r["lifetime_checks"] / 1000
        ops = "  ".join(f"{o} {v['accesses'] / k:.2f}" for o, v in c["by_operation"].items() if v["accesses"])
        rv = c["revokes"]
        print(f"  {name}: capability stores {tot / k:.1f}/1000 checks, same node {100 * st['index_unchanged_same_node'] / max(tot, 1):.1f}%,"
              f" index updated {100 * st['index_updated'] / max(tot, 1):.1f}%, revoked-cap stores {st['of_a_revoked_capability']:,};"
              f" data stores unregistering {c['data_stores_unregistering'] / k:.2f}/1000")
        print(f"      accesses by operation per 1000 checks: {ops}")
        print(f"      revokes {rv['count']:,}: accesses mean {rv['accesses_mean']:.1f}, max {rv['accesses_max']:,};"
              f" tag clears {rv['tag_clears']:,}")
        print(f"      alias records peak {c['alias_records_peak']:,} (x16 B = {c['alias_records_peak'] * 16 / 2**20:.1f} MiB),"
              f" sidecar frames peak {c['sidecar_frames_peak']:,} (x1 KiB = {c['sidecar_frames_peak'] / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
