#!/usr/bin/env python3
"""Compare a concurrent mix with the same programs run one after another.

    compare-mix.py <mix-par report.json> <mix-seq report.json> [ways]

Both runs execute the same programs on the same inputs in one boot each; they differ
only in whether the programs interleave on the hart. For every cache size the table
gives the lifetime-check (ldst + ldc) miss rate of each and the misses the
interleaving adds. ways: "full" (default), 1, 4 or 8. Exits non-zero if either report
has no lifetime checks.
"""
import json
import sys


def checks(c):
    h = c["by_site"]["ldst"][0] + c["by_site"]["ldc"][0]
    m = c["by_site"]["ldst"][1] + c["by_site"]["ldc"][1]
    return h, m


def main():
    par, seq = (json.load(open(p)) for p in sys.argv[1:3])
    ways = sys.argv[3] if len(sys.argv) > 3 else "full"
    ways = ways if ways == "full" else int(ways)
    rows = []
    for c in seq["caches"]:
        if c["ways"] != ways:
            continue
        p = next(x for x in par["caches"] if x["entries"] == c["entries"] and x["ways"] == ways)
        sh, sm = checks(c)
        ph, pm = checks(p)
        if not sh + sm or not ph + pm:
            print("ERROR: a report has no lifetime checks")
            return 1
        rows.append((c["entries"], sm, sh + sm, pm, ph + pm))
    print(f"lifetime checks: seq {rows[0][2]:,}  par {rows[0][4]:,}   ways={ways}")
    print(f"{'entries':>8} {'seq miss':>10} {'par miss':>10} {'par/seq':>8} {'added misses':>14}")
    for e, sm, st, pm, pt in rows:
        ratio = (pm / pt) / (sm / st) if sm else float("inf")
        print(f"{e:>8} {100*sm/st:9.3f}% {100*pm/pt:9.3f}% {ratio:8.2f} {pm - sm:>14,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
