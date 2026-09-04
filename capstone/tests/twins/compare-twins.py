#!/usr/bin/env python3
"""The agreement gate for the -O0/-O2 twins of a runtime suite.

Given the summary outputs of the same suite run at two optimisation levels, it
joins the per-benchmark verdicts and classifies each benchmark:

  AGREE-PASS     passed at both levels
  BOTH-FAIL      failed at both levels (a suite or runtime problem, not a level)
  <A>-ONLY-FAIL  failed at level A only   (e.g. O2-ONLY-FAIL: an optimisation miscompile)
  <B>-ONLY-FAIL  failed at level B only   (e.g. O0-ONLY-FAIL: a -O0 miscompile or a -O0-only crash)
  FLAKE          a side never booted (the runner said FLAKE): no verdict, rerun
  MISSING        the benchmark has a verdict on one side only

Exit 0 only when every benchmark is AGREE-PASS; 1 when any benchmark disagrees or
fails on both sides; 2 when a side has no verdicts at all, a benchmark is MISSING,
or a side flaked -- ABSENCE IS AN ERROR, never agreement.  An empty summary (the
suite died before reporting) must not read as "nothing failed".

Suite formats (the runners' own stdout, captured to summary.txt):
  rv8       "PASS  name" / "FAIL  name" / "SKIP  name"
  beebs     "run-all-beebs.sh: PASS[ (retried)] name (...)" / "FAIL name (...)" / "FLAKE name (...)"
  coremark  a single benchmark, PASS iff __COREMARK_PASSED__ is printed
"""
import argparse
import os
import re
import sys

# The runner prints "PASS  name" but "FAIL  name   (see <log glob>)": the verdict
# line may carry a trailing pointer, so the anchor is on the name, not the line end.
# The first run of the real suite reported an all-FAIL side as "no summary" because
# this anchored on end-of-line; the positive control had used a bare FAIL line.
RV8 = re.compile(r"^(PASS|FAIL|SKIP)\s+(\S+)(?:\s.*)?$")
BEEBS = re.compile(r"^run-all-beebs\.sh: (PASS|FAIL|FLAKE)\S*\s+(\S+)\b")


def parse(suite, path):
    """Return {bench: PASS|FAIL|FLAKE}; SKIP is not a verdict and is dropped."""
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as f:
        lines = f.read().replace("\r", "\n").split("\n")
    v = {}
    if suite == "rv8":
        for l in lines:
            m = RV8.match(l)
            if m and m.group(1) != "SKIP":
                v[m.group(2)] = m.group(1)
    elif suite == "beebs":
        for l in lines:
            m = BEEBS.match(l)
            if m:
                v[m.group(2)] = m.group(1)
    elif suite == "coremark":
        text = "\n".join(lines)
        # Any non-empty summary is a run that was attempted: the runner writes its own
        # failure lines to stderr, so a domain fault leaves only the build's lines on
        # stdout (W-17's jump-table arm, 2026-09-05, read as "no summary" instead of FAIL).
        # An EMPTY summary is still no verdict (the suite never ran).
        if text.strip():
            v["coremark"] = "PASS" if "__COREMARK_PASSED__" in text else "FAIL"
    else:
        raise SystemExit(f"unknown suite {suite}")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=["rv8", "beebs", "coremark"])
    ap.add_argument("--a", required=True, help="summary.txt of the first run")
    ap.add_argument("--b", required=True, help="summary.txt of the second run")
    ap.add_argument("--label-a", default="A", help="e.g. O0")
    ap.add_argument("--label-b", default="B", help="e.g. O2")
    ap.add_argument("--meta", default="", help="free text for every row (QEMU id, date)")
    ap.add_argument("--tsv", help="append one row per benchmark to this file")
    a = ap.parse_args()

    va, vb = parse(a.suite, a.a), parse(a.suite, a.b)
    rows, rc = [], 0
    if va is None or vb is None or not va or not vb:
        side = "both" if not va and not vb else (a.label_a if not va else a.label_b)
        rows.append((a.suite, "*", "ERROR", f"no verdicts at all on side {side}: the suite produced no summary"))
        rc = 2
    else:
        for bench in sorted(set(va) | set(vb)):
            x, y = va.get(bench), vb.get(bench)
            if x is None or y is None:
                verdict, detail, rc = "MISSING", f"{a.label_a}={x} {a.label_b}={y}", max(rc, 2)
            elif "FLAKE" in (x, y):
                verdict, detail, rc = "FLAKE", f"{a.label_a}={x} {a.label_b}={y}: never booted, rerun", max(rc, 2)
            elif x == "PASS" and y == "PASS":
                verdict, detail = "AGREE-PASS", ""
            elif x == "FAIL" and y == "FAIL":
                verdict, detail, rc = "BOTH-FAIL", "", max(rc, 1)
            elif x == "FAIL":
                verdict, detail, rc = f"{a.label_a}-ONLY-FAIL", "", max(rc, 1)
            else:
                verdict, detail, rc = f"{a.label_b}-ONLY-FAIL", "", max(rc, 1)
            rows.append((a.suite, bench, verdict, detail))

    out = []
    for suite, bench, verdict, detail in rows:
        line = "\t".join([f"twin {suite} {a.label_a}/{a.label_b} {a.meta}".strip(), bench, verdict, detail])
        out.append(line)
    print("\n".join(out))
    n = len(rows)
    agree = sum(1 for r in rows if r[2] == "AGREE-PASS")
    print(f"compare-twins: {a.suite} {a.label_a}/{a.label_b}: {agree}/{n} AGREE-PASS, exit {rc}")
    if a.tsv:
        os.makedirs(os.path.dirname(os.path.abspath(a.tsv)), exist_ok=True)
        with open(a.tsv, "a") as f:
            f.write("\n".join(out) + "\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
