#!/usr/bin/env python3
"""mstatus-read.py <uaf-repeat out dir>: the fault_mstatus readings of a uaf-repeat run (steps.json, written by uaf-repeat.py
with --dmesg). For every uaf run it takes the run's own fault cause and the module's refusal and refused-escape lines in that
run's dmesg tail ("capstone_vm: refused-escape mstatus 0x.. mprv N mpp N addr .. cause .."), and prints one line per run that
has any, then a tally of (run cause, mprv, mpp). A run with a refusal but NO mstatus line is printed as such (the print path
did not fire). Exit 0 when steps.json was read, 2 on no data. Written 2026-10-26 for the bitstream-3 scope note's step 0a."""
import collections
import json
import re
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    p = Path(sys.argv[1]) / "steps.json"
    if not p.is_file():
        print(f"NO DATA: {p}"); sys.exit(2)
    steps = json.loads(p.read_text())["steps"]
    tally = collections.Counter(); seen = set()
    for n, row in steps.items():
        t = row["text"]
        if not n.startswith("uaf-"):
            continue
        causes = re.findall(r"domain fault cause=(\d+)", t)
        dm = re.findall(r"^DM (.*)$", t, re.M)
        ref = [l for l in dm if "resolve refused" in l]
        ms = [l for l in dm if "refused-escape mstatus" in l]
        # The tail is the module's LAST 12 lines, so it carries earlier runs' lines too: only lines first seen in
        # this run (each carries its own kernel timestamp) belong to it.
        ref = [l for l in ref if l not in seen]
        ms = [l for l in ms if l not in seen]
        seen.update(dm)
        if not ref and not ms:
            continue
        print(f"RUN {n} cause={causes[0] if causes else None} refused={len(ref)} mstatus_lines={len(ms)}")
        for l in ms:
            m = re.search(r"mstatus (0x[0-9a-f]+) mprv (\d+) mpp (\d+) addr (0x[0-9a-f]+) cause (\d+)", l)
            print(f"  MS {l[:240]}")
            if m:
                tally[(causes[0] if causes else None, m.group(1), m.group(2), m.group(3))] += 1
        if ref and not ms:
            print("  MS NONE: a refusal with no refused-escape line")
    for (c, w, mprv, mpp), k in sorted(tally.items(), key=str):
        print(f"TALLY run-cause {c} mstatus {w} mprv {mprv} mpp {mpp}: {k}")
    if not tally:
        print("TALLY none: no refused-escape line in any run's dmesg tail")


if __name__ == "__main__":
    main()
