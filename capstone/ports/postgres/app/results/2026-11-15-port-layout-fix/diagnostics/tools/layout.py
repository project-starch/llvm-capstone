#!/usr/bin/env python3
"""layout.py ROOT...: sizeof and field offsets of the memory-context structures, by clang.

ROOT is a PG_SU_ROOT of a configured 17.5 build (ROOT/runtime holds the SDK's capstone-cc,
ROOT/domain/postgresql-17.5 the configured source). For aset.c, generation.c, slab.c and bump.c
this runs the build's own compile line (make -n -W, which remakes nothing) as clang's cc1 job
with -fsyntax-only -fdump-record-layouts, and prints the records of the structures that place memory.
"""
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

FILES = ("aset.c", "generation.c", "slab.c", "bump.c")
RECORDS = ("MemoryChunk", "AllocBlockData", "AllocSetContext", "GenerationBlock", "GenerationContext",
           "SlabBlock", "SlabContext", "BumpBlock", "BumpContext", "MemoryContextData")


def records(root):
    root = Path(root).resolve()
    mmgr = root / "domain/postgresql-17.5/src/backend/utils/mmgr"
    env = dict(os.environ, CAPSTONE_SDK=str(root / "runtime"),
               PATH=f"{root / 'runtime'}:{Path.home() / 'lanes/gatefix/llvm/bin'}:{os.environ['PATH']}")
    found = {}
    for name in FILES:
        obj = name[:-2] + ".o"
        cmd = subprocess.run(["make", "-n", "-W", name, obj], cwd=mmgr, env=env, capture_output=True,
                             text=True, check=True).stdout
        line = next(l for l in cmd.splitlines() if l.startswith("capstone-cc ") and f" {name}" in l)
        args = shlex.split(line)
        keep, skip = [], False
        for a in args:
            if skip:
                skip = False
                continue
            if a in ("-c",):
                continue
            if a in ("-o", "-MF"):
                skip = True
                continue
            if a in ("-MMD", "-MP"):
                continue
            keep.append(a)
        # capstone-cc passes no -Xclang option, so take its cc1 line (-###) and add the dump there.
        driver = subprocess.run(keep + ["-fsyntax-only", "-###"], cwd=mmgr, env=env, capture_output=True,
                                text=True, check=True).stderr
        cc1 = shlex.split(next(l for l in driver.splitlines() if '"-cc1"' in l))
        out = subprocess.run(cc1 + ["-fdump-record-layouts"], cwd=mmgr, env=env, capture_output=True,
                             text=True, check=True).stdout
        for block in out.split("*** Dumping AST Record Layout")[1:]:
            lines = [l.rstrip() for l in block.strip().splitlines()]
            m = re.match(r"\s*0 \| (?:struct|union) (\w+)", lines[0])
            if not m or m.group(1) not in RECORDS or m.group(1) in found:
                continue
            end = next(i for i, l in enumerate(lines) if "[sizeof=" in l)
            found[m.group(1)] = "\n".join(lines[:end + 1])
    return found


for root in sys.argv[1:]:
    print(f"=== {root}")
    got = records(root)
    for r in RECORDS:
        print(got.get(r, f"(no record {r})"))
        print()
