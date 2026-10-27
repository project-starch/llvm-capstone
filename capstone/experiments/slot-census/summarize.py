#!/usr/bin/env python3
"""Summarise one pass: the measured processes' census lines and each cell's validation verdict.

usage: summarize.py OUT_DIR [JSON_OUT]
A pass is a set of cells, each with a census file OUT_DIR/<cell>.jsonl while it runs (pack.py merges them into
OUT_DIR/processes.jsonl, one line per measured process with its cell). A pass with OUT_DIR/cells.tsv lists its cells
there as <cell> <selector> <verdict> [note], and a cell's measured processes are those its selector picks: an
executable's base name, re:<regex> searched in the whole command line, or spec. A SPEC pass (run-spec.sh) has no
cells.tsv until it is packed: one cell per benchmark, its verdict runcpu's in OUT_DIR/logs/<benchmark>.log. A SPEC
benchmark process (selector spec) is one whose executable name contains "_base.census" (SPEC names binaries
<exe>_base.<label>) and is not one of SPEC's image validators (imagevalidate_<n>, built with the same label, run on
511, 525, 526 and 538 outputs); the shells, specinvoke, specdiff and validator lines in the same file are dropped. 525.x264_r's ldecod_r, which decodes the input, stays and is reported as its own executable. Prints one
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


def spec(exe, cmdline):
    return is_benchmark(exe)


def keeper(selector):
    """a cells.tsv selector: an executable's base name, re:<regex> searched in the whole command line, or spec"""
    if selector == "spec":
        return spec
    if selector.startswith("re:"):
        rx = re.compile(selector[3:])
        return lambda exe, cmdline: rx.search(cmdline) is not None
    return lambda exe, cmdline: exe == selector


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
    """(cell, keep(executable, cmdline) -> bool, verdict) for every cell of the pass"""
    manifest = os.path.join(out, "cells.tsv")
    if os.path.exists(manifest):
        rows = [l.rstrip("\n").split("\t") for l in open(manifest) if "\t" in l]
        return [(c, keeper(x), v) for c, x, v, *_ in rows]
    benches = sorted(f[:-6] for f in os.listdir(out) if f.endswith(".jsonl"))
    return [(b, spec, verdict(os.path.join(out, "logs", b + ".log"))) for b in benches]


_merged = {}


def census_lines(out, cell):
    """a cell's census lines: from OUT_DIR/processes.jsonl in a packed pass, else from OUT_DIR/<cell>.jsonl"""
    merged = os.path.join(out, "processes.jsonl")
    if os.path.exists(merged):
        if merged not in _merged:
            _merged[merged] = {}
            for line in open(merged):
                _merged[merged].setdefault(json.loads(line)["cell"], []).append(line)
        return _merged[merged].get(cell, [])
    path = os.path.join(out, cell + ".jsonl")
    return open(path).readlines() if os.path.exists(path) else []


def processes(out, cell, keep):
    rows = []
    for line in census_lines(out, cell):
        r = json.loads(line)
        exe = os.path.basename(r["cmdline"].split(" ", 1)[0])
        if keep(exe, r["cmdline"]):
            r["exe"] = exe.split(MARK)[0]
            r["args"] = r["cmdline"].split(" ", 1)[1] if " " in r["cmdline"] else ""
            rows.append(r)
    return rows


def instruction_pass(out, todo):
    """a pass whose lines count instructions (slotins.c) rather than allocations"""
    for b, keep, v in todo:
        for line in census_lines(out, b):
            return "instructions" in json.loads(line)
    return False


def main():
    out = sys.argv[1]
    todo = cells(out)
    if not todo:
        print(f"summarize: no cells in {out}", file=sys.stderr)
        return 1
    ok, summary = True, []
    if instruction_pass(out, todo):   # a slotins pass (SLOTCENSUS_MEASURE=instructions): instructions per process
        print(f"{'cell':<24} {'executable':<24} {'instructions':>17} fork  arguments")
        for b, keep, v in todo:
            rows = processes(out, b, keep)
            ok &= bool(rows) and v == "valid"
            for r in rows:
                print(f"{b:<24} {r['exe']:<24} {r['instructions']:>17,} {r['forked']:>4}  {r['args'][:60]}")
            print(f"{b:<24} {'= sum (' + v + ')':<24} {sum(r['instructions'] for r in rows):>17,}" if rows
                  else f"{b:<24} NO DATA ({v})")
        print("SUMMARY OK" if ok else "SUMMARY INCOMPLETE: a cell lacks instruction lines or a valid verdict")
        return 0 if ok else 1
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
