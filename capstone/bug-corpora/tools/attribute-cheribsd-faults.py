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

The last column is the fault's live return-address register, resolved: it names the CALLER only when
the faulting function is a leaf (a probe, a libc memcpy); in a function that has made calls of its own it
is a stale return point. (Until 2026-10-11 the column was headed `caller`.)

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
# Printed by supervise.c since 2026-10-10: the mapped object holding the fault pc (and the return
# address), with that object's load base, so a fault in a shared library resolves against the
# library rather than reading as "outside every function" of the program.
OBJECT = re.compile(r"^SUPERVISE (fault-object|fault-ra) 0x([0-9a-f]+) in (\S+) base=0x([0-9a-f]+)\s*$", re.M)


def symbols(nm, program):
    """Defined functions; a stripped shared library falls back to its dynamic table."""
    table = {}
    for extra in ([], ["-D"]):
        out = subprocess.run([str(nm), "-S", "--defined-only", *extra, str(program)],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2] in "tTwWiI":
                # a versioned dynamic name (memcpy@@FBSD_1.0) is the function's plain name
                table.setdefault(parts[3].split("@", 1)[0], (int(parts[0], 16), int(parts[1], 16)))
        if table:
            break
    return table


def in_object(nm, elf, base, addr):
    """Every name of the function holding addr in an object loaded at base (a library exports one
    function under several aliases: memcpy, __memcpy, ...), from that object's own ELF."""
    hits = [(name, addr - base - value) for name, (value, size) in symbols(nm, elf).items()
            if size and value <= addr - base < value + size]
    if not hits:
        return "", addr - base
    return "|".join(sorted({n for n, _ in hits})), hits[0][1]


def resolve_object(text, which, program, sysroot):
    """(host ELF, base, addr) for the object a fault-object / fault-ra line names, or None. The
    program is staged in the guest as ./target or under its own name; a library maps to the sysroot."""
    for kind, addr, path, base in OBJECT.findall(text):
        if kind != which:
            continue
        name = path.rsplit("/", 1)[-1]
        if name in ("target", "program", pathlib.Path(program).name):
            return program, int(base, 16), int(addr, 16)
        if sysroot and (sysroot / path.lstrip("/")).is_file():
            return sysroot / path.lstrip("/"), int(base, 16), int(addr, 16)
        return None
    return None


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
    p.add_argument("--sysroot", type=pathlib.Path, help="the guest's sysroot, to resolve a fault in a shared library")
    a = p.parse_args()
    if not a.arm:
        p.error("name at least one --arm")
    sites = {k: set(v.split(",")) for k, v in (s.split("=", 1) for s in a.sites)}
    print("arm\tfault\tsi_code\tpc\tfunction\toffset\tdeclared_site_holds\treturn_address")
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
        lib = resolve_object(text, "fault-object", program, a.sysroot)
        if lib and str(lib[0]) != str(program):
            fn, off = in_object(a.nm, lib[0], lib[1], lib[2])
            fn = f"{fn}" if fn else ""
        caller = ""
        ra = resolve_object(text, "fault-ra", program, a.sysroot)
        if ra:
            cfn, coff = in_object(a.nm, ra[0], ra[1], ra[2])
            caller = f"{cfn or '?'}+{coff:#x} ({pathlib.Path(ra[0]).name})"
        if fn is None:
            print(f"{arm}\tSIGNAL {sig}\t{code}\t0x{pc}\tUNATTRIBUTED (no {a.anchor} anchor)\t\t")
            bad += 1
            continue
        holds = "" if arm not in sites else ("yes" if set(fn.split("|")) & sites[arm] else "NO")
        if holds == "NO" or not fn:
            bad += 1
        print(f"{arm}\tSIGNAL {sig}\t{code}\t0x{pc}\t{fn or 'outside every function'}\t{off:#x}\t{holds}\t{caller}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
