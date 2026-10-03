#!/usr/bin/env python3
"""Static audit: which instructions in a domain image would be ILLEGAL while that domain runs under supervision on
silicon (capstone-ariane 36a641e0b, the resident caplifive_supcall bitstream)?

Every rule below is the RTL's own guard, cited by file:line at 36a641e0b:
  CSR    csr_regfile.sv:2879-2886  plain CSR op (funct3 1/2/3/5/6/7) whose address has addr[9:8] != 0, or is one of
                                   0x800/0x801/0x802/0x804/0x810/0x811. (mcause 0x342, mtval 0x343, mcycle 0xB00 ...)
  CCSR   csr_regfile.sv:2880-2881  CCSRRW (custom-2 funct3 7) to CIH 0x001 or CPMP 0x010..0x01F
  PRIV   decoder.sv:232/269/289    SRET, MRET, WFI (SYSTEM funct3 0)
  CALL   decoder.sv:1289           custom-2 funct3 1 funct7 0x20 (no CALL under supervision)
  NEST   decoder.sv:1314           custom-2 funct3 1 funct7 0x22 (CSSUPERVISE)
  ENTER  decoder.sv:1318           custom-2 funct3 1 funct7 0x0D (CAPENTER)
  MINT   decoder.sv:1138           custom-3 funct3 0 funct7 0x04..0x08 (CAPCREATE/CAPTYPE/CAPNODE/CAPPERM/CAPBOUND)
and two that are not guards but end a supervised run the same way:
  VMOP   custom-2 funct3 1 funct7 0x23 -- the VM's collector opcode, NOT decoded on silicon at all
  TRAP   ECALL / EBREAK -- an exception under supervision is a fault event (kind 2), never a trap

Usage: sup-static-audit.py [--range LO:HI] <elf>...
Exit status: 0 = no forbidden instruction in any input; 1 = at least one found; 2 = no data (no instruction decoded,
objdump missing, unreadable input) -- an ERROR, never a clean result.
"""
import re
import subprocess
import sys
import os

OBJDUMP = os.environ.get("LLVM_OBJDUMP",
                         os.path.expanduser("~/dev/llvm-capstone/llvm/cmake-build-debug/bin/llvm-objdump"))
CSR_LIST = {0x800, 0x801, 0x802, 0x804, 0x810, 0x811}
CSR_NAMES = {0x300: "mstatus", 0x304: "mie", 0x305: "mtvec", 0x340: "mscratch", 0x341: "mepc", 0x342: "mcause",
             0x343: "mtval", 0x344: "mip", 0xB00: "mcycle", 0xB02: "minstret", 0xF14: "mhartid", 0x180: "satp",
             0x100: "sstatus", 0x141: "sepc", 0x142: "scause", 0x7C3: "csupquantum", 0x7C4: "csupctl",
             0xFC0: "csupstatus", 0xFC4: "csnodefree"}


def classify(w):
    """Return (rule, detail) if word w is forbidden under supervision, else None."""
    op = w & 0x7F
    f3 = (w >> 12) & 7
    f7 = w >> 25
    imm = w >> 20
    if op == 0x73:
        if f3 in (1, 2, 3, 5, 6, 7):
            if ((imm >> 8) & 3) != 0 or imm in CSR_LIST:
                return "CSR", f"csr 0x{imm:03x} ({CSR_NAMES.get(imm, '?')})"
            return None
        if f3 == 0:
            if w == 0x30200073:
                return "PRIV", "mret"
            if w == 0x10200073:
                return "PRIV", "sret"
            if w == 0x10500073:
                return "PRIV", "wfi"
            if w == 0x00000073:
                return "TRAP", "ecall"
            if w == 0x00100073:
                return "TRAP", "ebreak"
        return None
    if op == 0x5B:
        if f3 == 1:
            if f7 == 0x20:
                return "CALL", "domcall (funct7 0x20)"
            if f7 == 0x22:
                return "NEST", "cssupervise (funct7 0x22)"
            if f7 == 0x0D:
                return "ENTER", "capenter (funct7 0x0d)"
            if f7 == 0x23:
                return "VMOP", "VM-only funct7 0x23 (not decoded on silicon)"
        if f3 == 7:
            if imm == 0x001 or 0x010 <= imm <= 0x01F:
                return "CCSR", f"ccsrrw ccsr 0x{imm:03x} ({'cih' if imm == 1 else 'cpmp%d' % (imm - 0x10)})"
        return None
    if op == 0x7B and f3 == 0 and 0x04 <= f7 <= 0x08:
        return "MINT", f"custom-3 mint funct7 0x{f7:02x}"
    return None


LINE = re.compile(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(.*)$")
SYM = re.compile(r"^([0-9a-f]+) <(.+)>:$")


def audit(path, lo=None, hi=None):
    r = subprocess.run([OBJDUMP, "-d", "--triple=capstone64-unknown-elf", path], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR {path}: objdump rc={r.returncode}: {r.stderr.strip()[:200]}")
        return None
    hits, n, sym = [], 0, "?"
    for line in r.stdout.splitlines():
        m = SYM.match(line)
        if m:
            sym = m.group(2)
            continue
        m = LINE.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        bs = m.group(2).split()
        if len(bs) != 4:
            continue
        if lo is not None and not (lo <= addr < hi):
            continue
        n += 1
        w = int.from_bytes(bytes(int(b, 16) for b in bs), "little")
        c = classify(w)
        if c:
            hits.append((addr, sym, w, c[0], c[1], m.group(3).strip()))
    return n, hits


def main(argv):
    lo = hi = None
    if len(argv) > 2 and argv[1] == "--range":
        a, b = argv[2].split(":")
        lo, hi = int(a, 16), int(b, 16)
        argv = [argv[0]] + argv[3:]
    if len(argv) < 2:
        print(__doc__)
        return 2
    worst = 0
    for p in argv[1:]:
        res = audit(p, lo, hi)
        if res is None:
            worst = 2
            continue
        n, hits = res
        if n == 0:
            print(f"ERROR {p}: no 32-bit instruction decoded -- no data, not a clean result")
            worst = 2
            continue
        print(f"{p}: {n} instructions, {len(hits)} forbidden under supervision")
        for addr, sym, w, rule, det, dis in hits:
            print(f"  0x{addr:x} <{sym}> {w:08x} {rule:5s} {det}   [{dis}]")
        if hits and worst == 0:
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv))
