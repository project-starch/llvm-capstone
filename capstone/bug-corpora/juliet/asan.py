#!/usr/bin/env python3
"""A second opinion on MISSED cases, reported beside them and relabelling nothing: build each case's bad image natively
(x86-64 gcc, AddressSanitizer) and run it. REPORTED means ASan saw the defect natively; CLEAN that every run finished and
exited 0 without a report.

    asan.py --manifest M --juliet <dir>/C --verdicts <set>-verdicts.tsv --out asan.json [--control 200]

LIMITS, which is why judge.py does not relabel on it: a native run is not the guest's run (another libc, ABI and stack
layout), and ASan does not see accesses made inside an uninstrumented libc, e.g. glibc's wide-character printing.

SEEDS. A rand_flow case runs once per pattern of its first two rand()%2 results. glibc's rand() is not musl's, so the
native seeds are found by running a native rand() check (`native_seeds`), never copied from common.SEEDS.

POSITIVE CONTROL: a fixed sample (random seed 7) of the set's CAUGHT cases, each of which ASan should report. Its
per-case outcomes and rate are recorded; with no report at all the oracle is refused (exit 75).
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common as C

FLAGS = ["-O0", "-g", "-fsanitize=address", "-fno-omit-frame-pointer", "-DINCLUDEMAIN", "-DOMITGOOD"]


def native_seeds(work):
    src = work / "rc.c"
    src.write_text('#include <stdio.h>\n#include <stdlib.h>\nint main(void){for(unsigned s=1;s<=256;s++){srand(s);'
                   'int a=rand()%2,b=rand()%2;printf("%u %d%d\\n",s,a,b);}return 0;}\n')
    subprocess.run(["gcc", "-O0", str(src), "-o", str(work / "rc")], check=True)
    seeds = {}
    for line in subprocess.run([str(work / "rc")], capture_output=True, text=True, check=True).stdout.splitlines():
        s, pat = line.split()
        seeds.setdefault(pat, int(s))
    if sorted(seeds) != sorted(C.SEEDS):
        sys.exit(f"native rand() did not give all four patterns: {seeds}")
    return seeds


def run_case(c, juliet, work, seeds):
    tc, sup = juliet / "testcases", juliet / "testcasesupport"
    for pat, seed in (seeds.items() if seeds else [("plain", None)]):
        d = work / c["case"] / pat; d.mkdir(parents=True)
        srcs, found = [], 0
        for f in c["files"]:
            text = (tc / f).read_text(errors="replace")
            if seed is not None:
                text, n = C.overlay(text); found += n
            (d / Path(f).name).write_text(text); srcs.append(str(d / Path(f).name))
        if seed is not None and found != 1:
            return "OVERLAY-FAILED"
        exe = d / "bad"
        b = subprocess.run(["gcc", *FLAGS, *([f"-DJULIET_SEED={seed}"] if seed is not None else []), "-I", str(sup),
                            *srcs, str(sup / "io.c"), "-o", str(exe), "-lm"], capture_output=True, text=True)
        if b.returncode:
            return "BUILD-FAIL"
        try:
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL,
                               env=dict(os.environ, ASAN_OPTIONS="detect_leaks=0"))
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        out = r.stdout + r.stderr
        if "ERROR: AddressSanitizer:" in out:
            return "REPORTED"
        if not ("Finished bad()" in out and r.returncode == 0):
            return f"OTHER(rc={r.returncode})"
    return "CLEAN"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--juliet", type=Path, required=True)
    p.add_argument("--verdicts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--control", type=int, default=200)
    p.add_argument("--jobs", type=int, default=32)
    a = p.parse_args()
    cases = {c["case"]: c for c in C.load(a.manifest)["cases"]}
    with open(a.verdicts) as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [dict(zip(head, l.rstrip("\n").split("\t"))) for l in f]
    missed = [r["case"] for r in rows if r["verdict"] == "MISSED" or r["verdict"].startswith("NOT-TRIGGERED")]
    caught = sorted(r["case"] for r in rows if r["verdict"] == "CAUGHT")
    random.Random(7).shuffle(caught)
    control = sorted(caught[:a.control])
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(a.jobs) as ex:
        work = Path(tmp)
        seeds = native_seeds(work)
        pick = lambda k: seeds if cases[k]["rand_flow"] else None
        got = dict(zip(missed, ex.map(lambda k: run_case(cases[k], a.juliet, work / "m", pick(k)), missed)))
        ctl = dict(zip(control, ex.map(lambda k: run_case(cases[k], a.juliet, work / "c", pick(k)), control)))
    gcc = subprocess.run(["gcc", "-dumpfullversion"], capture_output=True, text=True).stdout.strip()
    rate = Counter(ctl.values())
    record = {"manifest_sha256": C.sha256(a.manifest), "compiler": f"gcc {gcc}", "flags": FLAGS, "native_seeds": seeds,
              "cases": got,
              "control": {"sample": len(ctl), "reported": rate.get("REPORTED", 0), "outcomes": dict(rate), "cases": ctl}}
    a.out.write_text(json.dumps(record, indent=1))
    print(f"control: {dict(rate)} of {len(ctl)}; missed: {dict(Counter(got.values()))}")
    sys.exit(0 if rate.get("REPORTED", 0) else 75)


if __name__ == "__main__":
    main()
