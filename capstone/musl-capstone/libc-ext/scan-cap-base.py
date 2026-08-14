#!/usr/bin/env python3
"""Flag `cincoffset`/`cincoffsetimm` whose BASE register holds an integer.

WHAT IT CATCHES. `cincoffset rd, rs1, rs2` requires the tagged capability in
rs1; the integer index belongs in rs2. When the optimiser splits a pointer
induction variable into a base and an index, the index is i128 on this target --
the same width as a capability -- and can be selected as capability arithmetic.
The result is `cincoffsetimm a1, a1, -1` on an `li a1, 12`, which QEMU rejects
with `helper_cscincoffsetimm: Assertion 'rs1_v->tag' failed` and silicon would
take as a capability fault.

Found 2026-08-14 in musl's vfprintf at -O1, in fmt_fp's digit loop
(`*--s = '0' + ...`), and the reason libc-ext's vfprintf is built at -O0. The
backend already canonicalises this for the cases it can see -- see
llvm/test/CodeGen/Capstone/cap-cincoffset-base.ll and isCapstoneCapabilityValue
-- so this is a gap in that classifier, not an absence of one.

WHAT IT IS NOT. A linear scan, not a dataflow analysis. For each site it walks
back to the NEAREST write of the base register, wherever that is, and reports
only if that writer is an integer instruction. Crossing basic blocks is
deliberate: the defect initialises the induction variable in the preheader and
increments it in the loop body. It can therefore miss a site whose nearest
writer is itself a laundered integer, and it does not prove absence. It proves
presence, which is what a build gate needs.

--self-test is a POSITIVE CONTROL and is run by the build before the real scan:
a two-case synthetic where exactly one must be flagged. A scanner that has never
flagged anything is not a passing scanner.
"""
import re
import sys

INT_OPS = {"li", "lui", "add", "addw", "addi", "addiw", "sub", "subw", "mul",
           "mulw", "mulh", "mulhu", "sll", "slli", "slliw", "srl", "srli",
           "srliw", "sra", "srai", "sraiw", "and", "andi", "or", "ori", "xor",
           "xori", "lw", "lwu", "ld", "lb", "lbu", "lh", "lhu", "sext.w",
           "zext.w", "neg", "not", "sllw", "srlw", "seqz", "snez", "slt",
           "sltu", "slti", "sltiu", "div", "divu", "rem", "remu", "auipc"}

INSN = re.compile(r"^\s+([a-z][a-z0-9.]*)\s+([a-z0-9]+)\s*(?:,\s*(.*))?$")

SELF_TEST = """
\tli\ta1, 12
\tcincoffsetimm\ta1, a1, -1
\tclc\ta2, 0(sp)
\tcincoffsetimm\ta2, a2, -1
""".splitlines()


def scan(lines):
    hits = []
    for i, line in enumerate(lines):
        m = INSN.match(line)
        if not m or m.group(1) not in ("cincoffset", "cincoffsetimm"):
            continue
        ops = [o.strip() for o in (m.group(2) + "," + (m.group(3) or "")).split(",")]
        if len(ops) < 2:
            continue
        base = ops[1]
        for j in range(i - 1, -1, -1):
            w = INSN.match(lines[j])
            if not w or w.group(2) != base:
                continue
            if w.group(1) in INT_OPS:
                hits.append((i + 1, line.strip(), j + 1, lines[j].strip()))
            break
    return hits


def main(argv):
    if not argv:
        print("usage: scan-cap-base.py [--self-test] <file.s>...", file=sys.stderr)
        return 2
    if argv[0] == "--self-test":
        hits = scan(SELF_TEST)
        if len(hits) != 1 or "a1" not in hits[0][1]:
            print(f"SELF-TEST FAILED, scanner is not working: {hits}", file=sys.stderr)
            return 2
        print("scan-cap-base: self-test ok (flags the li base, ignores the clc base)")
        return 0

    total = 0
    for path in argv:
        try:
            lines = open(path).read().splitlines()
        except OSError as exc:
            print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
            return 2
        if not lines:
            print(f"ERROR: {path} is empty; nothing was scanned", file=sys.stderr)
            return 2
        for line_no, insn, def_no, definition in scan(lines):
            total += 1
            print(f"{path}:{line_no}: {insn}\n    base defined at line {def_no}: {definition}")
    if total:
        print(f"{total} site(s) use an integer as a capability base", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
