#!/usr/bin/env python3
"""Find S-14 in a built domain image: a capability spill slot reloaded with a scalar `ld`.

WHY THIS EXISTS. S-14 was chased for two days as image geometry -- code size, globals count,
dom_data carve -- because those three moved together across every build anyone compared. They are
correlates. The mechanism is that `__capstone_cap_init` spills a capability with `stc` and reloads
it in at least one place with a plain `ld`, so the tag is gone and the next `cincoffsetimm` faults
with cause 24. Two triggers that looked unrelated -- restoring EXPLAIN, and enabling json -- are the
same defect.

GATE ON THIS, NOT ON A CARVE NUMBER. Changing the global set changes cap_init's codegen, so an image
can be clean today and faulty after an unrelated edit. A carve threshold would pass both.

EXITS NON-ZERO if it finds sites, AND if it cannot do its job -- an image with no
__capstone_cap_init, or an empty disassembly, is an ERROR and not a clean run. A "0" from a broken
scan reads exactly like a pass, which is the failure this project keeps paying for.

BLIND SPOTS, stated so a 0 is read correctly. It sees only `ld` from an `<imm>(sp)` slot; a reload
through another base register is invisible. It follows the loaded register for a few instructions
and stops at the first redefinition, so a longer dependence chain is invisible. And it counts only
`cincoffset*` as the consuming use, because that is what faults in the observed instances.
"""
import re, subprocess, sys, os

LOOKAHEAD = 8

def disassemble(dom, objdump):
    out = subprocess.run([objdump, "-d", "--triple=capstone64-unknown-elf", dom],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        sys.exit(f"capinit-reload-scan: objdump produced nothing for {dom}. "
                 f"Refusing to report a clean scan of an empty disassembly.")
    return out.stdout.split("\n")

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: capinit-reload-scan.py <domain.dom> [more.dom ...]")
    objdump = os.environ.get("CAPSTONE_OBJDUMP")
    if not objdump:
        build = os.environ.get("CAPSTONE_LLVM_BUILD_DIR")
        if not build:
            sys.exit("capinit-reload-scan: set CAPSTONE_LLVM_BUILD_DIR or CAPSTONE_OBJDUMP. "
                     "Refusing to guess the disassembler.")
        objdump = os.path.join(build, "bin", "llvm-objdump")
    rc = 0
    for dom in sys.argv[1:]:
        lines = disassemble(dom, objdump)
        if not any("__capstone_cap_init" in l for l in lines):
            print(f"{dom}: ERROR -- no __capstone_cap_init in this image; nothing was checked")
            rc = 2
            continue
        hits = []
        for i, l in enumerate(lines):
            m = re.search(r"^\s*([0-9a-f]+):.*\bld\s+(\w+), (0x[0-9a-f]+)\(sp\)", l)
            if not m:
                continue
            addr, reg, slot = m.group(1), m.group(2), m.group(3)
            for j in range(i + 1, min(i + 1 + LOOKAHEAD, len(lines))):
                if re.search(rf"\bcincoffset(imm)?\s+{reg}, {reg},", lines[j]):
                    hits.append((addr, reg, slot))
                    break
                if re.search(rf"\b(ld|ldc|li|mv|lw|lwu)\s+{reg},", lines[j]):
                    break
        if hits:
            rc = 1
            print(f"{dom}: {len(hits)} scalar reload(s) of a capability slot feeding cincoffset:")
            for addr, reg, slot in hits:
                print(f"   0x{addr}  ld {reg}, {slot}(sp)  ->  cincoffset {reg}")
            print("   The FIRST site is where the domain will fault (cap_init is straight-line).")
        else:
            print(f"{dom}: clean -- no scalar reload of a capability slot feeds a cincoffset")
    return rc

if __name__ == "__main__":
    sys.exit(main())
