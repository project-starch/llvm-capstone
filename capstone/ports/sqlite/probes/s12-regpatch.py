#!/usr/bin/env python3
"""Rewrite the register that carries the null across the S-12 fault window, and nothing else.

THE EXPERIMENT.  The S-12 window in the baseline SQLite domain is

    1047f0  movc          a4, zero
    1047f4  stc           a4, -0x5a0(s0)
    1047f8  sw            a4, 0x0(a5)
    1047fc  cincoffsetimm a5, s0, -0x110
    104800  sw            a4, 0x0(a5)
    104804  cincoffsetimm a5, s0, -0x120
    104808  movc          a4, zero
    10480c  stc           a4, 0x0(a5)
    104810  ldc           a4, 0x0(a0)
    104814  cincoffsetimm a4, a4, 0xb0      <- FAULTS: mcause 25, tval 0

This script changes the destination of the movc at 104808 and the source of the stc at 10480c
from a4 to a6, leaving every address, the instruction count, the distance to the fault, and the
store -- same address, same null value -- exactly as they were.  a6 is dead across the window and
a4 still holds zero from the movc at 1047f0, so the program means the same thing.  The only
variable is whether the register carrying the null matches the destination of the reload that
feeds the faulting consumer.

WHAT IT IS AND IS NOT EVIDENCE FOR.  An earlier round of this patch returned 7 times in 7 valid
board draws and was recorded as a cure at p ~ 1e-5.  That p-value was computed against a baseline
wedge rate of ~0.94 measured in a DIFFERENT slot configuration.  The null control for the arm
configuration gives one wedge in two valid draws, against which 7/7 is Fisher p ~ 0.22 -- not
significant.  So this patch is an untested candidate cure again, and the way to settle it is
s12stress.test (120 depth-2 prepares in one boot), not more 1-bit draws.

ANCHORING.  The window is located by its 20-byte instruction sequence, not by a hardcoded file
offset, and the script REFUSES to patch unless that sequence occurs exactly once.  A silently
mismatched anchor that patches the wrong place, or patches nothing and reports success, is the
failure this rule exists to prevent.
"""
import sys, hashlib

# cincoffsetimm a5,s0,-0x120 / movc a4,zero / stc a4,0(a5) / ldc a4,0(a0) / cincoffsetimm a4,a4,0xb0
ANCHOR = bytes.fromhex("db2704ee" "5b170014" "5bc0e700" "5b370500" "5b27070b")
A4, A6 = 14, 16


def set_field(word: int, shift: int, val: int) -> int:
    return (word & ~(0x1F << shift)) | (val << shift)


def main(src: str, dst: str) -> int:
    blob = bytearray(open(src, "rb").read())
    hits = []
    i = blob.find(ANCHOR)
    while i != -1:
        hits.append(i)
        i = blob.find(ANCHOR, i + 1)
    if len(hits) != 1:
        print(f"REFUSING: anchor occurs {len(hits)} times, need exactly 1", file=sys.stderr)
        return 2
    base = hits[0]

    movc = int.from_bytes(blob[base + 4:base + 8], "little")
    stc = int.from_bytes(blob[base + 8:base + 12], "little")
    if (movc >> 7) & 0x1F != A4 or (stc >> 20) & 0x1F != A4:
        print("REFUSING: register fields are not a4 -- the encoding assumption is wrong",
              file=sys.stderr)
        return 2

    blob[base + 4:base + 8] = set_field(movc, 7, A6).to_bytes(4, "little")   # movc rd
    blob[base + 8:base + 12] = set_field(stc, 20, A6).to_bytes(4, "little")  # stc  rs2

    changed = sum(a != b for a, b in zip(open(src, "rb").read(), blob))
    if changed != 3:
        print(f"REFUSING: {changed} bytes differ, expected exactly 3", file=sys.stderr)
        return 2

    open(dst, "wb").write(blob)
    print(f"patched at file offset 0x{base + 4:x}: movc/stc a4 -> a6, {changed} bytes changed")
    print(f"  in  {hashlib.sha256(open(src,'rb').read()).hexdigest()[:16]}")
    print(f"  out {hashlib.sha256(blob).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <baseline.dom> <patched.dom>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
