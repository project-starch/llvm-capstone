#!/usr/bin/env python3
"""Tabulate cachesim JSON reports.

    summarize.py report.json [report.json ...]

For each report: the access mix, the revoke walks, and the miss rate of every
simulated cache (rows: entries, columns: associativity). A report with no node
accesses is an error, not a table of zeros.
"""
import json
import sys


def pct(m, total):
    return f"{100.0 * m / total:7.3f}%" if total else "   n/a  "


def main():
    rc = 0
    for path in sys.argv[1:]:
        r = json.load(open(path))
        acc = r["accesses"]
        total = sum(sum(v.values()) for v in acc.values())
        print(f"== {path}")
        if total == 0:
            print("   ERROR: no node accesses in this trace")
            rc = 1
            continue
        print(f"   records {r['records']:,}  node accesses {total:,}  distinct nodes {r['distinct_nodes']:,}"
              f"  max id {r['max_node_id']:,}  resets {r['resets']}  nodes/line {r['nodes_per_line']}"
              + (f"  excluded {','.join(r['excluded_sites'])} ({r['excluded_records']:,})" if r["excluded_sites"] else ""))
        sites = list(acc["read"].keys())
        per_site = {s: sum(acc[k][s] for k in acc) for s in sites}
        print("   by site: " + "  ".join(f"{s} {n:,} ({100.0*n/total:.2f}%)" for s, n in per_site.items() if n))
        print("   by kind: " + "  ".join(f"{k} {sum(v.values()):,}" for k, v in acc.items()))
        nn = sum(r["checks_without_node"].values())
        if nn:
            print(f"   checks of capabilities with no node (no read): {nn:,}")
        comp = r.get("compulsory_misses")
        if comp:
            cm = sum(comp.values())
            cc = comp["ldst"] + comp["ldc"]
            print(f"   compulsory misses (any cache size): {cm:,} = {pct(cm, total).strip()} of accesses;"
                  f" of lifetime checks {cc:,}")
        w = r["revoke_walks"]
        if w["count"]:
            hist = "  ".join(f"{b}:{n:,}" for b, n in zip(w["hist_buckets"], w["hist"]) if n)
            print(f"   revokes {w['count']:,}  nodes invalidated {w['nodes_walked']:,}"
                  f"  mean {w['nodes_walked']/w['count']:.2f}  max {w['max']:,}   [{hist}]")
        ways = []
        for c in r["caches"]:
            if c["ways"] not in ways:
                ways.append(c["ways"])
        ways.sort(key=lambda x: 1 << 30 if x == "full" else x)
        grid = {(c["entries"], c["ways"]): c for c in r["caches"]}
        entries = sorted({c["entries"] for c in r["caches"]})
        print("   miss rate (all accesses)       " + "".join(f"{('%s-way' % w) if w != 'full' else 'full':>10}" for w in ways)
              + "   lifetime-check (ldst+ldc) miss rate, full")
        for e in entries:
            row = ""
            for w in ways:
                c = grid.get((e, w))
                row += f"{pct(c['misses'], c['hits'] + c['misses']) if c else '':>10}"
            f = grid[(e, "full")]
            ch = f["by_site"]["ldst"][0] + f["by_site"]["ldc"][0]
            cm = f["by_site"]["ldst"][1] + f["by_site"]["ldc"][1]
            print(f"   {e:>6} entries                 {row}   {pct(cm, ch + cm)}")
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
