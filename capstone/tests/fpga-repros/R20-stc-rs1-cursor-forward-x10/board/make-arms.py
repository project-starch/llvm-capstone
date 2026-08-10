#!/usr/bin/env python3
"""Regenerate the R-20 SQLite arms as byte patches of one base image.

The arms are NOT committed: each is 1.5 MB and differs from the base by 1-12 bytes, so shipping
them would be 9 MB of near-identical binaries. They are regenerated here and checked against
../SHA256SUMS, which is the actual frozen record.

  usage:  ./make-arms.py <base sqlite_silicon.dom> <outdir>

The base image is the "ar2" SQLite silicon build: sqlite3RegisterBuiltinFunctions clamped so that
only sqlite3AlterFunctions runs, with nDef = 1. Produce it with
capstone/benchmarks/sqlite/build-sqlite-silicon.sh; its sha256 is in ../SHA256SUMS.

Address arithmetic: the domain has a single PT_LOAD with .text at vaddr 0x10000 and file offset
0x1000, so file_offset = vaddr - 0xF000. Baking an arm is `cp -f` -- there is no checksum or
relink step inside the image.
"""
import hashlib
import pathlib
import sys

BASE_SHA = "5d8c27850f625f486afa1cef4df42b92fa7d222610380f0ba12fa6db70ce07ff"  # see ../SHA256SUMS
RET = 0x13CC68          # the function's OWN epilogue: ld ra / ldc s0 / cincoffsetimm sp / ret
A0, A3, S0 = 10, 13, 8


def jal(frm, to):
    """jal x0, to-frm  -- an unconditional jump, used to truncate a function early."""
    u = (to - frm) & 0x1FFFFF
    w = (((u >> 20) & 1) << 31) | (((u >> 1) & 0x3FF) << 21) | \
        (((u >> 11) & 1) << 20) | (((u >> 12) & 0xFF) << 12) | 0x6F
    return w.to_bytes(4, "little")


def branch(frm, to, funct3, rs1, rs2=0):
    u = (to - frm) & 0x1FFF
    w = ((((u >> 12) & 1) << 31) | (((u >> 5) & 0x3F) << 25) | (rs2 << 20) | (rs1 << 15) |
         (funct3 << 12) | (((u >> 1) & 0xF) << 8) | (((u >> 11) & 1) << 7) | 0x63)
    return w.to_bytes(4, "little")


def rtype(funct7, rs2, rs1, funct3, rd, op):
    return ((funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
            (rd << 7) | op).to_bytes(4, "little")


def itype(imm, rs1, funct3, rd, op):
    return (((imm & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) |
            (rd << 7) | op).to_bytes(4, "little")


def setfield(word, val, hi, lo):
    m = ((1 << (hi - lo + 1)) - 1) << lo
    return (word & ~m) | ((val << lo) & m)


NOP = (0x13).to_bytes(4, "little")


def build(base):
    """Return {name: [(vaddr, bytes), ...]}. Every arm is a patch of the SAME base image."""
    def word(va):
        o = va - 0xF000
        return int.from_bytes(base[o:o + 4], "little")

    arms = {}
    # --- the two headline arms -------------------------------------------------------------
    # Z: base with ONLY the branch offset changed (one byte, at 0x13cb6f). Both WEDGE, which is
    #    what proves the branch TARGET is irrelevant and removes the confound that every other
    #    arm changed target and position together.
    arms["Z"] = [(0x13CB6C, branch(0x13CB6C, RET, 0b000, A0))]
    # R13: the whole triple rewritten on a3 -- one variable versus base, the REGISTER NUMBER.
    #      Only the register fields of the real Capstone-format instructions are edited.
    arms["R13"] = [
        (0x13CB60, setfield(word(0x13CB60), A3, 11, 7).to_bytes(4, "little")),   # cincoffsetimm a3,s0,-0x50
        (0x13CB64, setfield(word(0x13CB64), A3, 19, 15).to_bytes(4, "little")),  # stc a1,0x0(a3)
        (0x13CB68, setfield(setfield(word(0x13CB68), A3, 11, 7), A3, 19, 15).to_bytes(4, "little")),
        (0x13CB6C, branch(0x13CB6C, RET, 0b000, A3)),
    ]
    # --- the value measurement -------------------------------------------------------------
    # V1: RETURN iff (a0 - s0) + 0x50 == 0, i.e. iff the value read for a0 was EXACTLY s0-0x50,
    #     the STC's own rs1 cursor. This MEASURES the poisoned value instead of inferring it.
    arms["V1"] = [
        (0x13CB6C, rtype(0x20, S0, A0, 0b000, A3, 0x33)),   # sub  a3, a0, s0
        (0x13CB70, itype(0x50, A3, 0b000, A3, 0x13)),       # addi a3, a3, 0x50
        (0x13CB74, branch(0x13CB74, RET, 0b000, A3)),       # beqz a3, RET
    ]
    # V0: the same chain with the read moved one slot later. Positive control -- it must RETURN
    #     via the OPPOSITE branch polarity, proving the chain can produce both answers.
    arms["V0"] = [
        (0x13CB6C, NOP),
        (0x13CB70, rtype(0x20, S0, A0, 0b000, A3, 0x33)),
        (0x13CB74, itype(0x50, A3, 0b000, A3, 0x13)),
        (0x13CB78, branch(0x13CB78, RET, 0b001, A3)),       # bnez a3, RET
    ]
    # --- the separation pair ---------------------------------------------------------------
    # gap: one nop between the LD and the branch.   adj: one nop between the STC and the LD.
    # Both RETURN. Breaking EITHER adjacency cures it.
    arms["gap"] = [(0x13CB6C, NOP), (0x13CB70, branch(0x13CB70, RET, 0b000, A0))]
    arms["adj"] = [(0x13CB68, NOP),
                   (0x13CB6C, base[0x13CB68 - 0xF000:0x13CB68 - 0xF000 + 4]),  # the ld, verbatim
                   (0x13CB70, branch(0x13CB70, RET, 0b000, A0))]
    # --- controls --------------------------------------------------------------------------
    # y1 truncates BEFORE the dereference and RETURNS; y2 truncates AFTER it and WEDGES. The pair
    # differs by exactly that one instruction and localises where execution stops.
    arms["y1"] = [(0x13CB7C, jal(0x13CB7C, RET))]
    arms["y2"] = [(0x13CB80, jal(0x13CB80, RET))]
    return arms


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    base = pathlib.Path(sys.argv[1]).read_bytes()
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    got = hashlib.sha256(base).hexdigest()
    print(f"base sha256 {got}")
    print(f"base bytes  {len(base)}")
    for name, patches in build(base).items():
        b = bytearray(base)
        for va, ins in patches:
            o = va - 0xF000
            assert len(ins) == 4, name
            b[o:o + 4] = ins
        blob = bytes(b)
        n_diff = sum(1 for i in range(len(blob)) if blob[i] != base[i])
        (out / f"{name}.dom").write_bytes(blob)
        print(f"  {name:4s} {hashlib.sha256(blob).hexdigest()[:16]}  {n_diff:2d} bytes differ from base")
    print(f"\nwrote {out}/ -- now verify against SHA256SUMS")


if __name__ == "__main__":
    main()
