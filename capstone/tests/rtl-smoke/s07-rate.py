#!/usr/bin/env python3
"""Classify every domain run in a board transcript and accumulate k/n.

WHY THIS EXISTS. The single most expensive methodological error on this project is reporting
"X wedges" from one run. The wedge is not deterministic and not a property of the image: the
SAME hash (`f1214600d0dac351`) has both passed and wedged on the same bitstream, the same
physical placement and the same position in the boot, eight minutes apart. And a wedge ENDS the
boot, so most wedges here are single samples by construction. The unit of evidence is therefore
k of n, and counting by hand is how it stops happening.

THREE OUTCOMES, and conflating the last two is the trap this exists to prevent:

  PASS        a result marker was emitted (`obs=` or a `__CAPSTONE_SQLITE_*_PASSED__` line).
  S07-WEDGE   the domain ENTERED (`SQ: G/enter`) and never returned. Counts toward k.
  NO-ENTRY    no entry marker: an entry stall or a host-side failure. Carries NO verdict about
              the code and must NOT count toward k or n. A run that never started is not a run
              that survived.

The full build reports via `__CAPSTONE_SQLITE_*_PASSED__` rather than `obs=`; the driver does
not know that and has mislabelled such a pass as "a truncation arm exiting early". Both forms
are accepted here.

usage:  s07-rate.py <transcript> [<transcript> ...]        # e.g. /tmp/capstone/sqlite-stages.txt
"""
import re
import sys

PASS_MARKERS = ("__CAPSTONE_SQLITE_EXTENDED_PASSED__", "__CAPSTONE_SQLITE_MEMORY_PASSED__",
                "__CAPSTONE_SQLITE_SILICON_PASSED__")


def classify(block):
    if any(m in block for m in PASS_MARKERS) or re.search(r"SQ: obs=\d+", block):
        return "PASS"
    if "SQ: G/enter" in block:
        return "S07-WEDGE"
    return "NO-ENTRY"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = []
    for path in sys.argv[1:]:
        try:
            text = open(path, errors="replace").read()
        except OSError as e:
            # A missing transcript is an ERROR, not an empty result: a silent skip would
            # under-count n and quietly inflate the rate.
            raise SystemExit(f"cannot read {path}: {e}")
        blocks = text.split("===== ")[1:]
        if not blocks:
            raise SystemExit(f"{path}: no per-domain blocks parsed -- that measured the parser, "
                             f"not the run. Point this at the run-scoped transcript "
                             f"(PROBE_SCOPED_OUT), not the driver log.")
        for b in blocks:
            name = b.split("=====")[0].strip().split("/")[-1].replace(".dom", "")
            dbas = re.search(r"DBAS:([0-9A-F]{8})", b)
            obs = re.search(r"SQ: obs=(\d+)", b)
            rows.append((path.split("/")[-1], name, classify(b),
                         dbas.group(1) if dbas else "-", obs.group(1) if obs else "-"))

    print(f"{'transcript':24s} {'domain':10s} {'outcome':10s} {'DBAS':9s} obs")
    for r in rows:
        print(f"{r[0]:24s} {r[1]:10s} {r[2]:10s} {r[3]:9s} {r[4]}")

    # Rate per domain, counting ONLY runs that actually started.
    print()
    by = {}
    for _, name, outcome, _, _ in rows:
        if outcome == "NO-ENTRY":
            continue
        k, n = by.get(name, (0, 0))
        by[name] = (k + (outcome == "S07-WEDGE"), n + 1)
    for name, (k, n) in sorted(by.items()):
        pct = f"{100.0 * k / n:.0f}%" if n else "-"
        print(f"  {name:10s} k={k} of n={n}   ({pct} wedged)")
    skipped = sum(1 for r in rows if r[2] == "NO-ENTRY")
    if skipped:
        print(f"  ({skipped} run(s) excluded as NO-ENTRY: never started, so they carry no verdict)")


if __name__ == "__main__":
    main()
