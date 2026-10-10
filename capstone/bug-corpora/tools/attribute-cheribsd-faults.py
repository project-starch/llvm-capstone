#!/usr/bin/env python3
"""Name the function a supervised CheriBSD fault landed in, from the run's own binary.

A CheriBSD arm run under `supervise` (cpython/pymalloc-repros/observe/supervise.c) prints, from
outside the child:

    SUPERVISE expect <anchor> 0x<runtime address of the anchor symbol>
    SUPERVISE fault signal=34 code=<si_code> addr=0x... pc=0x...

The anchor's runtime address minus its ELF value is the load base; the fault pc minus the base is
an ELF offset, and the defined function whose [value, value+size) holds it is where the fault
landed. That is the attribution the 2026-10-10 audits otherwise did by hand with llvm-nm.

    attribute-cheribsd-faults.py --run RUN --nm SDK/bin/llvm-nm --anchor ff2_case_run \\
        --arm so-00-buggy=BIN/so-00 [--arm ...] [--sites so-00-buggy=ff2_case_run ...] > attribution.tsv

RUN is run.py's output directory (one sub-directory per arm with stdout.txt). Prints one TSV row
per arm. Exit 0 when every arm was attributed (or completed without a fault); 1 when any fault
could not be attributed, or when a declared site does not hold the fault; 2 on bad input. A fault
with no anchor line is UNATTRIBUTED, never guessed.
"""
import argparse
import pathlib
import re
import subprocess
import sys

EXPECT = re.compile(r"^SUPERVISE expect (\S+) (?:0x([0-9a-f]+)|unavailable)\s*$", re.M)
FAULT = re.compile(r"^SUPERVISE fault signal=(\d+) code=(\S+) addr=(\S+) pc=0x([0-9a-f]+)\s*$", re.M)
EXIT = re.compile(r"^SUPERVISE exit (signalled|status)=(\d+)\s*$", re.M)


def symbols(nm, program):
    out = subprocess.run([str(nm), "-S", "--defined-only", str(program)],
                         capture_output=True, text=True, check=True).stdout
    table = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in "tTwW":
            table[parts[3]] = (int(parts[0], 16), int(parts[1], 16))
    return table


def attribute(table, anchor, anchor_rt, pc):
    if anchor_rt is None or anchor not in table:
        return None, None
    off = pc - (anchor_rt - table[anchor][0])
    for name, (value, size) in table.items():
        if size and value <= off < value + size:
            return name, off - value
    return "", off


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=pathlib.Path, required=True)
    p.add_argument("--nm", type=pathlib.Path, required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--arm", action="append", default=[], help="ARM=PROGRAM, repeatable")
    p.add_argument("--sites", action="append", default=[], help="ARM=fn1,fn2: declared fault sites, repeatable")
    a = p.parse_args()
    if not a.arm:
        p.error("name at least one --arm")
    sites = {k: set(v.split(",")) for k, v in (s.split("=", 1) for s in a.sites)}
    print("arm\tfault\tsi_code\tpc\tfunction\toffset\tdeclared_site_holds")
    bad = 0
    for spec in a.arm:
        arm, program = spec.split("=", 1)
        out = a.run / arm / "stdout.txt"
        if not out.is_file():
            print(f"{arm}\tNO-RECORD\t\t\t\t\t", flush=True)
            bad += 1
            continue
        text = out.read_text(errors="replace")
        faults = FAULT.findall(text)
        if not faults:
            ex = EXIT.search(text)
            print(f"{arm}\tnone ({ex.group(1)}={ex.group(2)})" if ex else f"{arm}\tnone (no exit line)", "\t\t\t\t\t", sep="")
            continue
        sig, code, _addr, pc = faults[0]
        m = EXPECT.search(text)
        anchor_rt = int(m.group(2), 16) if (m and m.group(1) == a.anchor and m.group(2)) else None
        fn, off = attribute(symbols(a.nm, program), a.anchor, anchor_rt, int(pc, 16))
        if fn is None:
            print(f"{arm}\tSIGNAL {sig}\t{code}\t0x{pc}\tUNATTRIBUTED (no {a.anchor} anchor)\t\t")
            bad += 1
            continue
        holds = "" if arm not in sites else ("yes" if fn in sites[arm] else "NO")
        if holds == "NO" or not fn:
            bad += 1
        print(f"{arm}\tSIGNAL {sig}\t{code}\t0x{pc}\t{fn or 'outside every function'}\t{off:#x}\t{holds}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
