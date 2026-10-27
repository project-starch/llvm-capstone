#!/usr/bin/env python3
"""Pack one pass into results/<stamp>/ as result lines with the host's paths removed.

usage: pack.py PASS_DIR DEST_DIR
Writes five files and nothing else:
  provenance.txt   the runner's provenance, and for a SPEC pass the runcpu config it generated (slotcensus.cfg)
  cells.tsv        <cell> <selector> <verdict> <note> per cell. A server pass's own manifest; for a SPEC pass one row
                   per benchmark with selector spec, runcpu's verdict and, as the note, its exit code and run time
                   and runcpu's Success:/Error: lines
  processes.jsonl  each cell's measured processes (see summarize.py), one census line each with its "cell" first
  outputs.txt      <cell> <line> for the program's own output where a server runner kept one (absent otherwise)
  control.txt      the controls' checker lines, one block per control (they carry both the expected and the
                   measured values, so the raw control records and expectation files are not kept)
The pass directory becomes <PASS> and the home directory <HOME>. Exit 1 if anything that looks like a home path
survives."""
import json
import os
import re
import sys

from summarize import cells


def main():
    src, dest = os.path.realpath(sys.argv[1]), sys.argv[2]
    home = os.path.realpath(os.path.expanduser("~"))
    scrub = lambda s: s.replace(src, "<PASS>").replace(home, "<HOME>")
    read = lambda *f: open(os.path.join(src, *f), errors="replace").read()
    os.makedirs(dest)
    write = lambda f, text: open(os.path.join(dest, f), "w").write(scrub(text))
    spec = not os.path.exists(os.path.join(src, "cells.tsv"))
    provenance = read("provenance.txt") if os.path.exists(os.path.join(src, "provenance.txt")) else ""
    if os.path.exists(os.path.join(src, "slotcensus.cfg")):
        provenance += "\n== slotcensus.cfg (the runcpu config this pass generated)\n" + read("slotcensus.cfg")
    write("provenance.txt", provenance)
    runs = {}   # a SPEC pass's runs.txt: <benchmark> rc=<n> <seconds>s
    if spec and os.path.exists(os.path.join(src, "runs.txt")):
        runs = dict(l.split(" ", 1) for l in read("runs.txt").splitlines() if " " in l)
    rows, census, outputs = [], [], []
    for cell, keep, verdict in cells(src):
        cmd = lambda l: json.loads(l)["cmdline"]
        path = os.path.join(src, cell + ".jsonl")
        for line in open(path) if os.path.exists(path) else []:
            if keep(os.path.basename(cmd(line).split(" ")[0]), cmd(line)):
                census.append(f'{{"cell": {json.dumps(cell)}, ' + line.lstrip()[1:])
        if os.path.exists(os.path.join(src, cell + ".out")):   # the program's own output (server runners)
            outputs += [f"{cell}\t{l}\n" for l in read(cell + ".out").splitlines() if l.strip()]
        if spec:
            log = os.path.join(src, "logs", cell + ".log")
            found = [l.strip() for l in open(log, errors="replace") if re.match(r"(Success|Error): ", l)] \
                if os.path.exists(log) else []
            note = " | ".join(x for x in [runs.get(cell, "").strip()] + found if x)
            rows.append("\t".join((cell, "spec", verdict, note)) + "\n")
    write("cells.tsv", "".join(rows) if spec else read("cells.tsv"))
    write("processes.jsonl", "".join(census))
    if outputs:
        write("outputs.txt", "".join(outputs))
    control = os.path.join(src, "control")
    write("control.txt", "".join(f"== {f[:-6]}\n" + read("control", f)
                                 for f in sorted(os.listdir(control)) if f.endswith(".check")))
    bad = []
    for f in sorted(os.listdir(dest)):
        for n, l in enumerate(open(os.path.join(dest, f), errors="replace"), 1):
            if re.search(r"/home/|/Users/|/root/", l):
                bad.append(f"{f}:{n}")
    if bad:
        print("pack: host paths survive:\n  " + "\n  ".join(bad[:20]), file=sys.stderr)
        return 1
    print(f"pack: {dest} written", file=sys.stderr)   # stderr: a caller may stream the pass as a tar on stdout
    return 0


if __name__ == "__main__":
    sys.exit(main())
