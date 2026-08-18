#!/usr/bin/env python3
"""Cross-check every capability store in __capstone_cap_init against the descriptor.

WHY. __capstone_cap_init (emitted by CapstoneCapGlobalInit.cpp) materialises pointer-valued
initialisers: for each one it loads the HOLDER's capability out of the gp cap-table and
stores the target capability into it at some byte offset. The entry glue carves each
holder's capability with the SIZE the descriptor records. So for a store of 16 bytes at
offset K into slot i, correctness requires

    K + 16 <= size(i)

Anything else is a store through a capability that is too small, which QEMU reports as

    Cap mem access OOB: cursor=... imm=32 size=16 bounds=(...,+16)

and which this silicon STALLS on instead of trapping (issue R-5), i.e. it presents as an
unexplained hang. Checking it statically costs seconds and names the exact global, versus
inferring it from a frozen pc.

The check is also the discriminator between the two candidate root causes, which is the
point: if the offsets are fine but the SIZES are wrong, the descriptor emitter is at fault;
if a store targets a slot whose size belongs to a NEIGHBOURING global, the code and the
table disagree about slot numbering.
"""
import os
import re
import struct
import subprocess
import sys


def descriptor(path):
    out = subprocess.run(["readelf", "-SW", path], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if ".capstone_gp_initdesc" not in line:
            continue
        f = line.replace("]", " ").split()
        off = int(f[5], 16)
        blob = open(path, "rb").read()
        _built, count = struct.unpack_from("<QQ", blob, off)
        return [struct.unpack_from("<QQq", blob, off + 32 + i * 24) for i in range(count)]
    raise SystemExit(f"{path}: no .capstone_gp_initdesc")


def objdump_bin():
    d = os.environ.get("CAPSTONE_LLVM_BIN")
    if d and os.path.isfile(os.path.join(d, "llvm-objdump")):
        return os.path.join(d, "llvm-objdump")
    guess = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "llvm", "cmake-build-debug", "bin", "llvm-objdump"))
    return guess if os.path.isfile(guess) else "llvm-objdump"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <domain.dom>")
    path = sys.argv[1]
    recs = descriptor(path)
    out = subprocess.run([objdump_bin(), "-d", "--triple=capstone64-unknown-elf", path],
                         capture_output=True, text=True).stdout

    in_fn = False
    regslot = {}     # reg -> slot index currently held (from ldc rd, imm(gp))
    imm = {}         # reg -> integer immediate (lui/addi chain)
    bad, checked = [], 0
    for line in out.splitlines():
        if line.startswith("0") and "<" in line:
            in_fn = "__capstone_cap_init" in line
            regslot.clear(); imm.clear()
            continue
        if not in_fn:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        mnem, ops = parts[-2].strip(), parts[-1].strip()

        # INVALIDATE ON ANY UNMODELLED WRITE. Without this, a register written by an
        # opcode this scanner does not understand keeps its stale tracked value and gets
        # attributed to a later gp access. That is not hypothetical: `slli a0, a0, 0xb`
        # feeding `cincoffset a0, gp, a0` was reported as "slot 235" purely from a stale
        # lui/addi chain, producing 7 confident, entirely fictional mismatches against a
        # 7-byte string literal. A scanner that cannot model an instruction must forget
        # its destination, never guess.
        MODELLED = ("lui", "addi", "add", "cincoffset", "ldc", "stc")
        if mnem not in MODELLED and ops:
            dst = ops.split(",")[0].strip()
            if re.fullmatch(r"[a-z]\w*", dst):
                imm.pop(dst, None)
                regslot.pop(dst, None)

        if mnem == "lui" and ops.count(",") == 1:
            d, v = (x.strip() for x in ops.split(","))
            try:
                x = (int(v, 0) << 12) & 0xFFFFFFFF
                imm[d] = x - (1 << 32) if x >= (1 << 31) else x
            except ValueError:
                imm.pop(d, None)
        elif mnem in ("addi", "add") and ops.count(",") == 2:
            d, s, v = (x.strip() for x in ops.split(","))
            try:
                imm[d] = imm.get(s, 0) + int(v, 0)
            except ValueError:
                imm.pop(d, None)
        # TRACK THE LEVEL OF INDIRECTION, not just the slot. There are three:
        #   A  `cincoffset rd, gp, X`  -> rd addresses the TABLE ENTRY for slot X/16
        #   B  `ldc rd, 0(A)`          -> rd is the HOLDER capability (bounds = that
        #                                 global's size); stores through it are checkable
        #   C  `ldc rd, 0(B)`          -> rd is a capability STORED INSIDE the global and
        #                                 points at some other object entirely
        # Conflating B and C makes every `stc` through a loaded pointer look like a store
        # into the holder, which reported 7 bogus mismatches against a 7-byte string.
        elif mnem == "cincoffset" and ", gp," in ops:
            d, _gp, s = (x.strip() for x in ops.split(","))
            if s in imm and imm[s] >= 0:
                regslot[d] = (imm[s] // 16, "A", 0)
            else:
                regslot.pop(d, None)
        elif mnem in ("cincoffset", "cincoffsetimm") and ops.count(",") == 2:
            # A HOLDER OFFSET BY A REGISTER OR IMMEDIATE IS STILL THE HOLDER.
            #
            # This case was MISSING and it is the one that mattered. The compiler does not
            # always fold the byte offset into the `stc` immediate: for large offsets it
            # materialises the offset in a register (lui/addi) and does
            # `cincoffset rd, <holder>, rN`, then stores at 0(rd). The old code matched only
            # the `, gp,` form here, so `rd` kept a STALE entry or none, and the `stc` below
            # took the `continue` path -- the store was silently NOT CHECKED.
            #
            # That is how this script returned "OK: every capability store fits" for a domain
            # that faults with CAPABLITY_OUT_OF_BOUND inside __capstone_cap_init. A checker
            # that skips the failing store reads exactly like a checker that passed it.
            d, s, o = (x.strip() for x in ops.split(","))
            base = regslot.get(s)
            delta = imm.get(o) if not o.lstrip("-").startswith(("0x", "0", "1", "2", "3", "4",
                                                               "5", "6", "7", "8", "9")) else None
            if delta is None:
                try:
                    delta = int(o, 0)
                except ValueError:
                    delta = imm.get(o)
            if base is not None and base[1] == "B" and delta is not None:
                regslot[d] = (base[0], "B", base[2] + delta)
            else:
                regslot.pop(d, None)
        elif mnem == "ldc":
            m = re.match(r"(\w+),\s*(-?0x[0-9a-f]+|\d+)\((\w+)\)", ops)
            if not m:
                continue
            d, off, base = m.group(1), int(m.group(2), 0), m.group(3)
            if base == "gp":
                regslot[d] = (off // 16, "B", 0)  # folded ldc rd, imm(gp): holder in hand
            elif base in regslot and off == 0 and regslot[base][1] == "A":
                regslot[d] = (regslot[base][0], "B", 0)
            else:
                regslot.pop(d, None)            # level C or unknown: stop tracking
        elif mnem == "stc":
            m = re.match(r"(\w+),\s*(-?0x[0-9a-f]+|\d+)\((\w+)\)", ops)
            if not m:
                continue
            off, base = int(m.group(2), 0), m.group(3)
            entry = regslot.get(base)
            if entry is None or entry[1] != "B":
                continue                        # only a holder capability is checkable
            slot = entry[0]
            if slot >= len(recs):
                continue
            off += entry[2]          # offset folded into a preceding cincoffset
            checked += 1
            size = recs[slot][0]
            if off + 16 > size:
                bad.append((line.strip(), slot, off, size))

    print(f"descriptor records : {len(recs)}")
    print(f"stores checked     : {checked}")
    if not bad:
        print("OK: every capability store fits its holder's declared size")
        return 0
    print(f"MISMATCH: {len(bad)} store(s) exceed the holder size the descriptor declares")
    for line, slot, off, size in bad[:15]:
        print(f"  slot {slot}: store at +{off} needs {off + 16} bytes, descriptor says "
              f"{size}\n      {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
