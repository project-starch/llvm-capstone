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


def text(src, *path):
    p = os.path.join(src, *path)
    return open(p, errors="replace").read() if os.path.exists(p) else ""


def provenance(src):
    """the runner's provenance, and a SPEC pass's generated runcpu config"""
    cfg = text(src, "slotcensus.cfg")
    return text(src, "provenance.txt") + (
        "\n== slotcensus.cfg (the runcpu config this pass generated)\n" + cfg if cfg else "")


def spec_row(src, cell, verdict, runs):
    """a SPEC benchmark's cells.tsv row: its note is the exit code and run time (runs.txt) and runcpu's
    Success:/Error: lines"""
    found = [l.strip() for l in text(src, "logs", cell + ".log").splitlines() if re.match(r"(Success|Error): ", l)]
    note = " | ".join(x for x in [runs.get(cell, "").strip()] + found if x)
    return "\t".join((cell, "spec", verdict, note)) + "\n"


def measured_lines(src, cell, keep):
    """the cell's census lines of the processes its selector measures, each with "cell" first"""
    cmd = lambda line: json.loads(line)["cmdline"]
    return [f'{{"cell": {json.dumps(cell)}, ' + line.lstrip()[1:]
            for line in text(src, cell + ".jsonl").splitlines(keepends=True)
            if keep(os.path.basename(cmd(line).split(" ")[0]), cmd(line))]


def control_checks(src):
    """every control's checker lines, one block per control"""
    control = os.path.join(src, "control")
    return "".join(f"== {f[:-6]}\n" + text(control, f) for f in sorted(os.listdir(control)) if f.endswith(".check"))


def host_paths(dest):
    """file:line of every line in DEST that still names a home directory"""
    return [f"{f}:{n}" for f in sorted(os.listdir(dest))
            for n, l in enumerate(open(os.path.join(dest, f), errors="replace"), 1)
            if re.search(r"/home/|/Users/|/root/", l)]


def main():
    src, dest = os.path.realpath(sys.argv[1]), sys.argv[2]
    home = os.path.realpath(os.path.expanduser("~"))
    os.makedirs(dest)
    write = lambda f, t: open(os.path.join(dest, f), "w").write(t.replace(src, "<PASS>").replace(home, "<HOME>"))
    spec = not os.path.exists(os.path.join(src, "cells.tsv"))
    runs = dict(l.split(" ", 1) for l in text(src, "runs.txt").splitlines() if " " in l)   # SPEC: rc=<n> <s>s
    rows, census, outputs = [], [], []
    for cell, keep, verdict in cells(src):
        census += measured_lines(src, cell, keep)
        outputs += [f"{cell}\t{l}\n" for l in text(src, cell + ".out").splitlines() if l.strip()]
        if spec:
            rows.append(spec_row(src, cell, verdict, runs))
    write("provenance.txt", provenance(src))
    write("cells.tsv", "".join(rows) if spec else text(src, "cells.tsv"))
    write("processes.jsonl", "".join(census))
    if outputs:
        write("outputs.txt", "".join(outputs))
    write("control.txt", control_checks(src))
    bad = host_paths(dest)
    if bad:
        print("pack: host paths survive:\n  " + "\n  ".join(bad[:20]), file=sys.stderr)
        return 1
    print(f"pack: {dest} written", file=sys.stderr)   # stderr: a caller may stream the pass as a tar on stdout
    return 0


if __name__ == "__main__":
    sys.exit(main())
