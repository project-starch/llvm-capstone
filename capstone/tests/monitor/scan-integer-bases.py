#!/usr/bin/env python3
"""scan-integer-bases.py <firmware.elf|disassembly> [--lo 0x80020000] [--hi 0x80040000]

Count the plain loads and stores whose BASE REGISTER was last written by an INTEGER op -- the accesses
that become NOT_CAP faults once the LSU's capability check is delivered (R-34's fix) in M-mode with
capmode set. Splits the count by address range: the monitor's capability-mode text (`--lo`..`--hi`)
against everything else, because the two have different owners and different exposure.

POSITIVE CONTROL, and it is not optional: the monitor's hand-written trap site in `_handle_non_ecall`
(`add t5, t5, sp` then `sd a0, 0x10(t5)`) MUST appear in the output. A first version of this scan
restricted itself to functions that adjust their frame with a capability instruction and so excluded
that 13-instruction fragment, reporting a confident ZERO with the one known site missing. If the
control line below does not print, the scan is not measuring what it claims and its zero means nothing.
"""
import re, sys, subprocess, collections

INT = {"add","addi","addw","addiw","sub","subw","slli","srli","srai","or","ori","and","andi","xor","xori",
       "mv","li","lui","auipc","sh1add","sh2add","sh3add","sext.w","zext.w","neg","seqz","snez","slt",
       "sltu","slti","sltiu","mul","mulw","div","divu","rem","remu"}
CAP = {"cincoffsetimm","cincoffset","ldc","movc","scc","ccsrrw","capenter","split","shrink","shrinkto",
       "mrev","delin","tighten","seal","gencap","captype","capbound","capperm","capcreate"}
LS  = {"ld","lw","lwu","lh","lhu","lb","lbu","sd","sw","sh","sb"}
INSN = re.compile(r"^\s*([0-9a-f]+):\s+[0-9a-f ]+\t([a-z][a-z0-9._]*)\s*\t?(.*)$")
MEM  = re.compile(r"^(-?0x[0-9a-f]+|-?\d+)\((\w+)\)$")

def disassemble(path):
    if path.endswith(".dis"):
        return open(path, errors="replace").read().splitlines()
    import os
    objdump = os.environ.get("CAPSTONE_LLVM_BIN", "") + "/llvm-objdump"
    out = subprocess.run([objdump, "-d", "--triple=capstone64-unknown-elf", path],
                         capture_output=True, text=True, errors="replace")
    if out.returncode != 0:
        sys.exit(f"disassembly failed: set CAPSTONE_LLVM_BIN (capstone-test-env.sh). {out.stderr[:200]}")
    return out.stdout.splitlines()

def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    lo = int(sys.argv[sys.argv.index("--lo")+1], 16) if "--lo" in sys.argv else 0x80020000
    hi = int(sys.argv[sys.argv.index("--hi")+1], 16) if "--hi" in sys.argv else 0x80040000
    func, start, code = None, {}, collections.defaultdict(list)
    for l in disassemble(sys.argv[1]):
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:", l)
        if m:
            func = m.group(2); start[func] = int(m.group(1), 16); continue
        mm = INSN.match(l)
        if mm and func: code[func].append((mm.group(2), mm.group(3)))
    tot, bad, per = collections.Counter(), collections.Counter(), collections.Counter()
    control = 0
    for f, ins in code.items():
        if f == "payload_bin": continue          # the embedded kernel image, not code we own
        region = "cap_text" if lo <= start.get(f, 0) < hi else "other"
        last = {}
        for mnem, ops in ins:
            if mnem in LS:
                parts = [p.strip() for p in ops.split(",")]
                mo = MEM.match(parts[-1]) if len(parts) >= 2 else None
                if mo:
                    tot[region] += 1
                    if last.get(mo.group(2)) == "int":
                        bad[region] += 1; per[(region, f)] += 1
                        if f == "_handle_non_ecall": control += 1
            d = ops.split(",")[0].strip()
            if mnem in INT and d: last[d] = "int"
            elif mnem in CAP and d: last[d] = "cap"
    for r in ("cap_text", "other"):
        print(f"{r:9} {tot[r]:6} plain accesses, {bad[r]:6} through an integer-derived base")
    print(f"POSITIVE CONTROL _handle_non_ecall: {control} (MUST be >= 1, or this scan proves nothing)")
    for (r, f), n in per.most_common(15):
        print(f"   {r:9} {n:5}  {f}")
    sys.exit(0 if control >= 1 else 2)

main()
