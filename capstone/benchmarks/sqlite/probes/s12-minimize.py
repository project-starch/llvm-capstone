#!/usr/bin/env python3
"""Minimise S-12 by removing INSTRUCTIONS from the real faulting binary.

Not a reconstruction and not a source truncation. Both of those have now failed for the same
reason: they regenerate the function, so every address, allocation and instruction count moves at
once and the result is a different program that has to be argued about. Twelve hand-written
reconstructions came back clean, and truncating the source made SQLite return a garbage Bitmask
that killed the run in memcpy long before the site under test.

This patches the ACTUAL binary that faults, replacing chosen instructions with NOP
(addi x0,x0,0 = 0x00000013) and leaving every other byte, address and offset identical. What
survives removal is not needed; what cannot be removed is the repro.

The window, entry to fault, is 36 instructions and branch-free -- the same path on every call:

    [14] cincoffsetimm a0, s0, -0x70    a0 = the spill slot
    [16] stc  a2, 0x0(a0)               store the incoming pWInfo into it
    [34] ldc  a4, 0x0(a0)               reload it
    [35] cincoffsetimm a4, a4, 0xb0     FAULTS, mcause 25, tval 0

[34] and [35] are the fault itself and are refused as targets. [0]-[4] establish the frame. The
other 27 are candidates, and bisecting them costs ~log2(27) rounds rather than 27.

This also answers, directly and without statistics, the question the arm comparisons could not:
NOP [32] and [33] -- the `movc a4,zero` / `stc a4,0(a5)` pair -- and if the fault persists, the
pair was never required. The whole register-match correlation, which has been retracted twice and
re-derived twice, is one boot away from being settled by deletion instead of by p-values.

USAGE
    s12-minimize.py <in.dom> <out.dom> --nop 5,6,7,8      remove instructions by index
    s12-minimize.py <in.dom> --list                       print the window with indices

The anchor is the 20-byte sequence around the fault, required to occur exactly once, so a drifted
offset refuses rather than patching the wrong place.
"""
import argparse
import hashlib
import re
import subprocess
import sys

OBJDUMP = "/home/alexey/dev/llvm-capstone/llvm/cmake-build-debug/bin/llvm-objdump"
TRIPLE = "capstone64-unknown-elf"
FUNC = "sqlite3WhereCodeOneLoopStart"
NOP = bytes.fromhex("13000000")          # addi x0, x0, 0, little-endian
FAULT_VA = 0x104814
# cincoffsetimm a5,s0,-0x120 / movc a4,zero / stc a4,0(a5) / ldc a4,0(a0) / cincoffsetimm a4,a4,0xb0
ANCHOR = bytes.fromhex("db2704ee" "5b170014" "5bc0e700" "5b370500" "5b27070b")


def window(path):
    """[(index, va, text)] from function entry through the faulting instruction."""
    out = subprocess.run([OBJDUMP, "-d", f"--triple={TRIPLE}", path],
                         capture_output=True, text=True, timeout=600)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit(f"objdump produced nothing for {path}")
    started, rows = False, []
    for line in out.stdout.splitlines():
        if re.match(rf"^([0-9a-f]+)\s+<{FUNC}>:", line):
            started = True
            continue
        if not started:
            continue
        m = re.match(r"\s+([0-9a-f]+):\s+((?:[0-9a-f]{2} ){4})\s*(.*)", line)
        if not m:
            continue
        va = int(m.group(1), 16)
        rows.append((len(rows), va, m.group(3).strip()))
        if va == FAULT_VA:
            return rows
    sys.exit(f"fault site 0x{FAULT_VA:x} not reached in {FUNC}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--nop", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    rows = window(a.src)
    if a.list:
        for i, va, txt in rows:
            tag = ""
            if i in (34, 35):
                tag = "   <- the fault itself, REFUSED as a target"
            elif i <= 4:
                tag = "   <- frame setup"
            print(f"  [{i:2d}] 0x{va:x}  {txt}{tag}")
        return 0

    if not a.dst:
        sys.exit("need <out.dom> unless --list")
    idx = sorted({int(x) for x in a.nop.split(",") if x.strip() != ""})
    if not idx:
        sys.exit("--nop needs at least one index")
    bad = [i for i in idx if i in (34, 35)]
    if bad:
        sys.exit(f"REFUSING: {bad} is the faulting pair itself; removing it removes the bug, "
                 f"not the cause")
    if any(i < 0 or i >= len(rows) for i in idx):
        sys.exit(f"REFUSING: index out of range 0..{len(rows)-1}")

    blob = bytearray(open(a.src, "rb").read())
    hits = []
    p = blob.find(ANCHOR)
    while p != -1:
        hits.append(p)
        p = blob.find(ANCHOR, p + 1)
    if len(hits) != 1:
        sys.exit(f"REFUSING: anchor occurs {len(hits)} times, need exactly 1")
    # The anchor starts at instruction [31]; derive the file offset of [0] from it rather than
    # trusting any recorded constant.
    base = hits[0] - 31 * 4

    for i in idx:
        off = base + i * 4
        blob[off:off + 4] = NOP
    changed = sum(x != y for x, y in zip(open(a.src, "rb").read(), blob))
    if changed > 4 * len(idx):
        sys.exit(f"REFUSING: {changed} bytes differ, expected at most {4*len(idx)}")

    open(a.dst, "wb").write(blob)
    print(f"NOPed {len(idx)} instruction(s) {idx} at file offset 0x{base:x}; "
          f"{changed} bytes changed")
    print(f"  in  {hashlib.sha256(open(a.src,'rb').read()).hexdigest()[:16]}")
    print(f"  out {hashlib.sha256(blob).hexdigest()[:16]}")

    after = window(a.dst)
    if after[35][2] != rows[35][2] or after[34][2] != rows[34][2]:
        sys.exit("REFUSING: the fault pair changed; the patch landed in the wrong place")
    print(f"  fault pair intact: [34] {after[34][2]} / [35] {after[35][2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
