#!/usr/bin/env python3
"""Copy one run-spec.sh pass into results/<stamp>/ as raw lines with the host's paths removed.

usage: pack.py PASS_DIR DEST_DIR
Copies the control files, each cell's census lines (its measured processes only, see summarize.py) and the program's
own output where a server runner kept one, provenance.txt, runs.txt or cells.tsv, and for a SPEC pass the generated
config and runcpu's Success:/Error: lines from each benchmark log (verdicts.txt). The pass
directory becomes <PASS> and the home directory <HOME>. Exit 1 if anything that looks like a home path survives."""
import json
import os
import re
import sys

from summarize import cells


def main():
    src, dest = os.path.realpath(sys.argv[1]), sys.argv[2]
    home = os.path.realpath(os.path.expanduser("~"))
    scrub = lambda s: s.replace(src, "<PASS>").replace(home, "<HOME>")
    copy = lambda f: open(os.path.join(dest, f), "w").write(scrub(open(os.path.join(src, f), errors="replace").read()))
    os.makedirs(os.path.join(dest, "control"))
    for f in sorted(os.listdir(os.path.join(src, "control"))):
        copy(os.path.join("control", f))
    for f in ("slotcensus.cfg", "provenance.txt", "runs.txt", "cells.tsv"):
        if os.path.exists(os.path.join(src, f)):
            copy(f)
    spec = not os.path.exists(os.path.join(src, "cells.tsv"))
    verdicts = []
    for cell, keep, _ in cells(src):
        exe = lambda l: os.path.basename(json.loads(l)["cmdline"].split(" ")[0])
        path = os.path.join(src, cell + ".jsonl")
        lines = [l for l in open(path) if keep(exe(l))] if os.path.exists(path) else []
        open(os.path.join(dest, cell + ".jsonl"), "w").write(scrub("".join(lines)))
        if os.path.exists(os.path.join(src, cell + ".out")):   # the program's own output (server runners)
            copy(cell + ".out")
        if spec:
            log = os.path.join(src, "logs", cell + ".log")
            found = []
            if os.path.exists(log):
                found = [l for l in open(log, errors="replace") if re.match(r"(Success|Error): ", l)]
            verdicts.append(f"{cell}: " + ("".join(found).strip().replace("\n", " | ") or "no verdict line") + "\n")
    if spec:
        open(os.path.join(dest, "verdicts.txt"), "w").write(scrub("".join(verdicts)))
    bad = []
    for root, _, files in os.walk(dest):
        for f in files:
            for n, l in enumerate(open(os.path.join(root, f), errors="replace"), 1):
                if re.search(r"/home/|/Users/|/root/", l):
                    bad.append(f"{os.path.join(root, f)}:{n}")
    if bad:
        print("pack: host paths survive:\n  " + "\n  ".join(bad[:20]), file=sys.stderr)
        return 1
    print(f"pack: {dest} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
