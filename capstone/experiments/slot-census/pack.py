#!/usr/bin/env python3
"""Copy one run-spec.sh pass into results/<stamp>/ as raw lines with the host's paths removed.

usage: pack.py PASS_DIR DEST_DIR
Copies the control files, each benchmark's census lines (benchmark processes only, see summarize.py), the generated
config, provenance.txt, runs.txt, and runcpu's Success:/Error: lines from each benchmark log (verdicts.txt). The pass
directory becomes <PASS> and the home directory <HOME>. Exit 1 if anything that looks like a home path survives."""
import json
import os
import re
import sys

from summarize import is_benchmark


def main():
    src, dest = os.path.realpath(sys.argv[1]), sys.argv[2]
    home = os.path.realpath(os.path.expanduser("~"))
    scrub = lambda s: s.replace(src, "<PASS>").replace(home, "<HOME>")
    os.makedirs(os.path.join(dest, "control"))
    for f in sorted(os.listdir(os.path.join(src, "control"))):
        open(os.path.join(dest, "control", f), "w").write(scrub(open(os.path.join(src, "control", f)).read()))
    for f in ("slotcensus.cfg", "provenance.txt", "runs.txt"):
        open(os.path.join(dest, f), "w").write(scrub(open(os.path.join(src, f)).read()))
    verdicts = []
    for f in sorted(os.listdir(src)):
        if not f.endswith(".jsonl"):
            continue
        exe = lambda l: os.path.basename(json.loads(l)["cmdline"].split(" ")[0])
        keep = [l for l in open(os.path.join(src, f)) if is_benchmark(exe(l))]
        open(os.path.join(dest, f), "w").write(scrub("".join(keep)))
        log = os.path.join(src, "logs", f[:-6] + ".log")
        lines = []
        if os.path.exists(log):
            lines = [l for l in open(log, errors="replace") if re.match(r"(Success|Error): ", l)]
        verdicts.append(f"{f[:-6]}: " + ("".join(lines).strip().replace("\n", " | ") or "no verdict line") + "\n")
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
