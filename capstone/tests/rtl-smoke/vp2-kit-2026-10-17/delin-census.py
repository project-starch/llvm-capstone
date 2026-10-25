#!/usr/bin/env python3
"""delin-census.py <image.dom>...: count the `delin`s whose operand was just derived from gp.

vp2 Tier 1/2 boot 2 (2026-10-18) faulted with cause 26 at the image-gp global-address materialization,
`scc rd, gp, x; delin rd`. The virtual ABI's gp is NONLIN, the RTL raises UNEXPECTED_CAP_TYPE on DELIN of any
non-LINEAR capability, and QEMU before the strict-DELIN pin returned early. This counts, per image, every `delin rd`
whose rd was last written, within the WINDOW preceding instructions, by cincoffset, cincoffsetimm or scc with gp as
rs1. The gate on rebuilt images is 0. The images boot 2 ran are the positive control: contract.dom bf433e10 525 of
530, sqlite3.dom a4b2c0f0 5,983 of 5,988, mruby.dom 5fee8d8f 9,859 of 9,863.

Exit 0 when every image reads 0 gp-derived delins. Exit 1 when any reads more. Exit 2 on no data: an image that
disassembles to no instruction, or a missing objdump. "No data" is never a zero.
Totals are printed beside the count, so a change in the codegen's shape (a delin moved out of the window) shows as
a total that did not drop by the count.

  LLVM_OBJDUMP=<path> python3 delin-census.py a.dom b.dom      (default: the repo's llvm/cmake-build-debug build)
"""
import os, re, subprocess, sys

WINDOW = 5
DERIVE = ("cincoffset", "cincoffsetimm", "scc")
INSN = re.compile(r"\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} ){4}\s*(\S+)\s*(.*)")


def census(lines):
    """(instructions, delins, gp-derived delins, first sites) over llvm-objdump -d output lines."""
    ins = [(m.group(1), m.group(2), m.group(3)) for m in map(INSN.match, lines) if m]
    delins = gp = 0
    sites = []
    for i, (addr, op, args) in enumerate(ins):
        if op != "delin":
            continue
        delins += 1
        rd = args.split(",")[0].strip()
        for j in range(i - 1, max(i - 1 - WINDOW, -1), -1):
            regs = [r.strip() for r in ins[j][2].split(",")]
            if regs and regs[0] == rd:
                if ins[j][1] in DERIVE and len(regs) > 1 and regs[1] == "gp":
                    gp += 1
                    if len(sites) < 3:
                        sites.append(f"{ins[j][0]}: {ins[j][1]} {ins[j][2]}; {addr}: delin {rd}")
                break
    return len(ins), delins, gp, sites


def selftest():
    t = ["   10: 5b 95 a1 0a  \tscc\ta0, gp, a0", "   14: 5b 15 00 06  \tdelin\ta0",
         "   18: 5b 96 a1 0a  \tcincoffset\ta1, sp, a0", "   1c: 5b 16 00 06  \tdelin\ta1",
         "   20: 5b 95 a1 0a  \tscc\ta2, gp, a0", "   24: 13 06 06 00  \tmv\ta2, a3", "   28: 5b 15 00 06  \tdelin\ta2"]
    got = census(t)[:3]
    assert got == (7, 3, 1), got   # only the first: a1 comes from sp, and a2 is redefined by mv before its delin
    return True


def main(argv):
    selftest()
    od = os.environ.get("LLVM_OBJDUMP", os.path.expanduser("~/dev/llvm-capstone/llvm/cmake-build-debug/bin/llvm-objdump"))
    if not os.access(od, os.X_OK):
        print(f"NO DATA: no llvm-objdump at {od}"); return 2
    if not argv:
        print(__doc__); return 2
    worst = 0
    for f in argv:
        r = subprocess.run([od, "-d", "--triple=capstone64-unknown-elf", f], capture_output=True, text=True)
        n, d, g, sites = census(r.stdout.splitlines())
        sha = subprocess.run(["sha256sum", f], capture_output=True, text=True).stdout[:16]
        if n == 0:
            print(f"NO DATA {f} ({sha}): 0 instructions disassembled (objdump rc {r.returncode})"); worst = 2; continue
        print(f"{'FAIL' if g else 'ok  '} {f} ({sha}): {g} gp-derived delin of {d} delin, {n} instructions")
        for s in sites:
            print(f"       {s}")
        if g and worst < 1:
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
