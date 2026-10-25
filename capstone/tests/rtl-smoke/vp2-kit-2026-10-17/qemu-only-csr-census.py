#!/usr/bin/env python3
"""qemu-only-csr-census.py <image or module>...: count accesses to CSRs that QEMU implements and the vp2 RTL does not.

vp2 Tier 1 boot 3-T1 (2026-10-18) faulted with cause 2 at `malloc`'s `csrr a0, 0xcc0`. CSR 0xcc0 is urevavail
(ISSUES Q-21), and 0x5c0 is scapctl (Q-22). QEMU implements both; on the RTL a read or write is an illegal instruction.
So every QEMU record passed while the board faulted at the first allocation. This counts every Zicsr instruction
(csrrw, csrrs, csrrc and their immediate forms) whose CSR field is a QEMU-only CSR, in each file's code.
- Capability images (.dom) are disassembled with llvm-objdump --triple=capstone64-unknown-elf.
- Kernel modules (.ko) are disassembled with the buildroot cross objdump.
Exit 0 when every file reads 0. Exit 1 when any reads more. Exit 2 on no data: a file that disassembles to no
instruction, or a missing objdump. "No data" is never a zero.
Positive control: boot 3-T1's images (contract.dom f2017064, sqlite3.dom 233640f7, mruby bba891bc) read 2 each, in
malloc and realloc. Negative control: capstone_vm.ko ed24f827 reads 0 (Q-22's answer: the silicon adapter is built
without them).

  python3 qemu-only-csr-census.py a.dom b.dom m.ko
"""
import os, re, subprocess, sys

QEMU_ONLY = {0xcc0: "urevavail (Q-21)", 0x5c0: "scapctl (Q-22)"}
LLVM = os.environ.get("LLVM_OBJDUMP", os.path.expanduser("~/dev/llvm-capstone/llvm/cmake-build-debug/bin/llvm-objdump"))
CROSS = os.environ.get("CROSS_OBJDUMP", os.path.expanduser(
    "~/dev/llvm-capstone/capstone/caplifive-system/sw/buildroot/build-fpga/host/bin/riscv64-buildroot-linux-gnu-objdump"))


def words(lines):
    """(function, 32-bit word) for each 4-byte instruction, from either objdump's -d output."""
    fn = None
    for l in lines:
        h = re.match(r"[0-9a-f]+ <([^>]*)>:", l)
        if h:
            if not h.group(1).startswith("."): fn = h.group(1)
            continue
        m = re.match(r"\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} ){4})\s", l)        # llvm-objdump: four byte pairs
        if m:
            yield fn, int.from_bytes(bytes.fromhex(m.group(1).replace(" ", "")), "little"); continue
        m = re.match(r"\s*[0-9a-f]+:\s+([0-9a-f]{8})\s", l)                  # GNU objdump: one 8-digit word
        if m:
            yield fn, int(m.group(1), 16)


def census(lines):
    n, hits = 0, []
    for fn, w in words(lines):
        n += 1
        if w & 0x7F == 0x73 and (w >> 12) & 7 in (1, 2, 3, 5, 6, 7) and (w >> 20) in QEMU_ONLY:
            hits.append((fn, w >> 20))
    return n, hits


def selftest():
    llvm_like = ["0000000000020000 <malloc>:", "   20ccc: 73 25 00 cc  \tcsrr\ta0, 0xcc0",
                 "   20cd0: 73 25 10 c0  \trdtime\ta0", "   20cd4: 73 10 00 5c  \tcsrw\t0x5c0, zero"]
    gnu_like = ["0000000000000000 <f>:", "   0:\tcc002573          \tcsrr\ta0,0xcc0", "   4:\t00000013          \tnop"]
    n, h = census(llvm_like); assert (n, sorted(c for _, c in h)) == (3, [0x5c0, 0xcc0]), (n, h)
    n, h = census(gnu_like); assert (n, [c for _, c in h]) == (2, [0xcc0]), (n, h)


def main(argv):
    selftest()
    if not argv:
        print(__doc__); return 2
    worst = 0
    for f in argv:
        od = CROSS if f.endswith(".ko") else LLVM
        if not os.access(od, os.X_OK):
            print(f"NO DATA: no objdump at {od}"); return 2
        args = [od, "-d", f] if od == CROSS else [od, "-d", "--triple=capstone64-unknown-elf", f]
        r = subprocess.run(args, capture_output=True, text=True)
        n, hits = census(r.stdout.splitlines())
        sha = subprocess.run(["sha256sum", f], capture_output=True, text=True).stdout[:16]
        if n == 0:
            print(f"NO DATA {f} ({sha}): 0 instructions disassembled (objdump rc {r.returncode})"); worst = 2; continue
        by = {}
        for fn, c in hits: by.setdefault(f"{c:#x} {QEMU_ONLY[c]}", set()).add(fn or "?")
        print(f"{'FAIL' if hits else 'ok  '} {f} ({sha}): {len(hits)} QEMU-only CSR accesses in {n} instructions"
              + "".join(f"; {k} in {', '.join(sorted(v))}" for k, v in sorted(by.items())))
        if hits and worst < 1: worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
