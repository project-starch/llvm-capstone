#!/usr/bin/env python3
"""Summarise one pass: the measured processes' census lines and each cell's validation verdict.

usage: summarize.py OUT_DIR [JSON_OUT]
A pass is a set of cells, each with a census file OUT_DIR/<cell>.jsonl. A pass with OUT_DIR/cells.tsv (the server
runners) lists its cells there as <cell> <executable> <verdict>, and a cell's measured processes are those of that
executable. A SPEC pass (run-spec.sh) has one cell per benchmark, its verdict is runcpu's in
OUT_DIR/logs/<benchmark>.log (or, once packed, OUT_DIR/verdicts.txt), and a benchmark process is one whose executable name contains "_base.census" (SPEC
names binaries <exe>_base.<label>) and is not one of SPEC's image validators (imagevalidate_<n>, built with the same
label, run on 511, 525, 526 and 538 outputs); the shells, specinvoke, specdiff and validator lines in the same file
are dropped. 525.x264_r's ldecod_r, which decodes the input, stays and is reported as its own executable. Prints one
row per benchmark process (an invocation of one input) and one per cell (the maximum over its processes, which is
what sizes the slots), and with JSON_OUT writes the same as JSON there (derived data stays out of the pass
directory). Exit 1 if any cell has no census line or no valid verdict: "no data" is an error, never a zero."""
import json
import os
import re
import sys

MARK = "_base.census"


def is_benchmark(exe):
    return MARK in exe and not exe.startswith("imagevalidate_")


def verdict(log):
    try:
        return verdict_text(open(log, errors="replace").read())
    except FileNotFoundError:
        return "no log"


def verdict_text(text):
    text = text.replace(" | ", "\n")
    if re.search(r"^Success: ", text, re.M) and not re.search(r"^Error: ", text, re.M):
        return "valid"
    if re.search(r"^Error: ", text, re.M) or "*** Miscompare" in text:
        return "INVALID"
    return "no verdict"


def cells(out):
    """(cell, keep(executable) -> bool, verdict) for every cell of the pass"""
    manifest = os.path.join(out, "cells.tsv")
    if os.path.exists(manifest):
        rows = [l.rstrip("\n").split("\t") for l in open(manifest) if "\t" in l]
        return [(c, lambda e, x=x: e == x, v) for c, x, v, *_ in rows]
    benches = sorted(f[:-6] for f in os.listdir(out) if f.endswith(".jsonl"))
    packed = os.path.join(out, "verdicts.txt")   # a pass packed by pack.py keeps runcpu's verdict lines only
    if not os.path.isdir(os.path.join(out, "logs")) and os.path.exists(packed):
        lines = dict(l.rstrip("\n").split(": ", 1) for l in open(packed))
        return [(b, is_benchmark, verdict_text(lines.get(b, ""))) for b in benches]
    return [(b, is_benchmark, verdict(os.path.join(out, "logs", b + ".log"))) for b in benches]


def processes(out, cell, keep):
    rows = []
    path = os.path.join(out, cell + ".jsonl")
    for line in open(path) if os.path.exists(path) else []:
        r = json.loads(line)
        exe = os.path.basename(r["cmdline"].split(" ", 1)[0])
        if keep(exe):
            r["exe"] = exe.split(MARK)[0]
            r["args"] = r["cmdline"].split(" ", 1)[1] if " " in r["cmdline"] else ""
            rows.append(r)
    return rows


def main():
    out = sys.argv[1]
    todo = cells(out)
    if not todo:
        print(f"summarize: no cells in {out}", file=sys.stderr)
        return 1
    ok, summary = True, []
    print(f"{'cell':<24} {'executable':<24} {'peak_live':>11} {'allocs':>13} {'peak_bytes':>14} fork  arguments")
    for b, keep, v in todo:
        rows = processes(out, b, keep)
        if not rows or v != "valid":
            ok = False
        for r in rows:
            print(f"{b:<24} {r['exe']:<24} {r['peak_live']:>11,} {r['allocs']:>13,} {r['peak_live_bytes']:>14,} "
                  f"{r['forked']:>4}  {r['args'][:60]}")
        top = max(rows, key=lambda r: r["peak_live"]) if rows else None
        if top:
            print(f"{b:<24} {'= max (' + v + ')':<24} {top['peak_live']:>11,} {sum(r['allocs'] for r in rows):>13,} "
                  f"{max(r['peak_live_bytes'] for r in rows):>14,}")
        else:
            print(f"{b:<24} NO DATA ({v})")
        summary.append({"cell": b, "validation": v, "processes": len(rows),
                        "peak_live": top["peak_live"] if top else None,
                        "peak_live_bytes": max((r["peak_live_bytes"] for r in rows), default=None),
                        "allocs": sum(r["allocs"] for r in rows),
                        "invocations": [{k: r[k] for k in ("exe", "args", "forked", "peak_live", "end_live", "allocs",
                                                           "frees", "realloc_moved", "realloc_inplace", "failed",
                                                           "peak_live_bytes", "size_log2")} for r in rows]})
    if len(sys.argv) > 2:
        json.dump(summary, open(sys.argv[2], "w"), indent=1)
    print("SUMMARY OK" if ok else "SUMMARY INCOMPLETE: a cell lacks census lines or a valid verdict")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
