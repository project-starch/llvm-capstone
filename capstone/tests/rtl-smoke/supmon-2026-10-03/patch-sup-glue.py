#!/usr/bin/env python3
"""Make a silicon-ABI domain image runnable under supervision on silicon, by replacing ONLY the words that
sup-static-audit.py flags, each with a fixed equivalent:

  csrw  mcause, t0          (glue reentry: restore the saved cause)  -> nop
  csrw  mtval,  t0          (glue reentry)                           -> nop
  csrrw t0, mcause, zero    (glue return: save and clear the cause)  -> li t0, 0
  csrrw t0, mtval,  zero    (glue return)                            -> li t0, 0
  csrr  rd, mcycle   0xB00  (cycle bracket)                          -> csrr rd, cycle   0xC00 (the same counter)
  csrr  rd, minstret 0xB02                                           -> csrr rd, instret 0xC02

Why the first four are neutral under a PLAIN call: the monitor reads mcause only at trap entry and in the
_dom_reentry spin (sbi_capstone.S:4-9, :69), never after __domcallsaves returns. Under supervision the switch's
SAVE/RESTORE walks restore the monitor's own CSRs anyway (supervised-call-silicon.md).

Refuses (exit 1) if any flagged word has no substitution, if the output differs from the input anywhere else,
or if the audit of the output is not clean. Usage: patch-sup-glue.py <in.dom> <out.dom>
"""
import hashlib
import importlib.util
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("audit", os.path.join(HERE, "sup-static-audit.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

NOP, LI_T0_0 = 0x00000013, 0x00000293
FIXED = {0x34229073: NOP, 0x34329073: NOP, 0x342012F3: LI_T0_0, 0x343012F3: LI_T0_0}


def substitute(w):
    if w in FIXED:
        return FIXED[w]
    csr, f3, rs1 = w >> 20, (w >> 12) & 7, (w >> 15) & 31
    if (w & 0x7F) == 0x73 and f3 == 2 and rs1 == 0 and csr in (0xB00, 0xB02):  # csrr rd, mcycle/minstret
        return (w & 0x000FFFFF) | ((csr + 0x100) << 20)
    return None


def va_to_off(elf, va):
    phoff, = struct.unpack_from("<Q", elf, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", elf, 0x36)
    for i in range(phnum):
        p_type, _, p_off, p_va, _, p_filesz, _, _ = struct.unpack_from("<IIQQQQQQ", elf, phoff + i * phentsize)
        if p_type == 1 and p_va <= va < p_va + p_filesz:
            return p_off + va - p_va
    raise SystemExit(f"REFUSED: VA 0x{va:x} is in no PT_LOAD segment")


def main(src, dst):
    elf = bytearray(open(src, "rb").read())
    res = audit.audit(src)
    if res is None or res[0] == 0:
        raise SystemExit("REFUSED: the audit decoded nothing (no data)")
    n, hits = res
    for addr, sym, w, rule, det, dis in hits:
        new = substitute(w)
        if new is None:
            raise SystemExit(f"REFUSED: no substitution for 0x{addr:x} <{sym}> {w:08x} {rule} {det}")
        off = va_to_off(elf, addr)
        old, = struct.unpack_from("<I", elf, off)
        if old != w:
            raise SystemExit(f"REFUSED: word at 0x{addr:x} (file 0x{off:x}) is {old:08x}, objdump said {w:08x}")
        struct.pack_into("<I", elf, off, new)
        print(f"  0x{addr:x} <{sym}> {w:08x} -> {new:08x}  ({det})")
    orig = open(src, "rb").read()
    diff = sum(1 for i in range(0, len(orig), 4) if orig[i:i + 4] != bytes(elf[i:i + 4]))
    if diff != len(hits):
        raise SystemExit(f"REFUSED: {diff} words differ, {len(hits)} substitutions made")
    open(dst, "wb").write(elf)
    os.chmod(dst, 0o755)
    n2, hits2 = audit.audit(dst)
    if n2 != n or hits2:
        raise SystemExit(f"REFUSED: output audit {n2} instructions, {len(hits2)} forbidden")
    print(f"{src} sha256 {hashlib.sha256(orig).hexdigest()[:16]} -> {dst} sha256 "
          f"{hashlib.sha256(elf).hexdigest()[:16]}: {len(hits)} words substituted, {n2} instructions, audit clean")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
