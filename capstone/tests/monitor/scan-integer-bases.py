#!/usr/bin/env python3
"""scan-integer-bases.py <firmware.elf|disassembly> [--lo 0x80020000] [--hi 0x80040000]

Count the plain loads and stores whose BASE REGISTER was last written by an INTEGER op -- the accesses
that become NOT_CAP faults once the LSU's capability check is delivered (R-34's fix) in M-mode with
capmode set. Splits the count by address range: the monitor's capability-mode text (`--lo`..`--hi`)
against everything else, because the two have different owners and different exposure.

POSITIVE CONTROL, and it is not optional. A first version of this scan restricted itself to functions
that adjust their frame with a capability instruction, so it excluded the 13-instruction hand-written
fragment holding the one known site and reported a confident ZERO. Without a control, a zero here means
nothing.

THE CONTROL IS THE `other` REGION, NOT `_handle_non_ecall`, AND THAT CHANGED 2026-09-17. It used to be
the trap site itself, which was the one known instance -- and that made the scan SELF-DEFEATING: the
D3 fix removes exactly that site, so on a CORRECT firmware the control could never fire and the scan
exited 2, i.e. it could not be used to verify the fix it was built to find, and its failure would read
as a broken instrument rather than as a repaired monitor. Demonstrated with a matched pair on the same
firmware, one site changed: before, `cap_text` 1 and exit 0; after, `cap_text` 0 and exit 2.

The generic OpenSBI text carries ~7,115 such accesses and nobody is fixing them, so it fires whatever
the monitor's capability-mode text does. It is independent of the subject, which is what a control has
to be.

EXIT CODES: 2 the control did not fire, so the scan proves nothing; 1 the control fired and `cap_text`
is NON-ZERO, which is the finding this scan exists to report; 0 the control fired and `cap_text` is
zero, which is the only combination that means the monitor is clean AND the check was working.
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

def die(msg):
    """Every CANNOT-CHECK path exits 2, never 1.

    `sys.exit(<string>)` exits 1, and so does an uncaught exception -- which is the SAME code this
    scan uses for "the control fired and the capability-mode text is non-zero", i.e. the finding.
    So a broken invocation was indistinguishable from a positive result to any caller reading the
    status rather than the output. Reported from a real run 2026-09-17: with CAPSTONE_LLVM_BIN
    unset, subprocess raises before the return-code guard below is ever reached, the process exits
    1, and that reads as the finding. Three paths had it -- the missing binary, the guard itself,
    and an unreadable input file.
    """
    print(f"BLOCKED: {msg}", file=sys.stderr)
    print("BLOCKED: this scan could not measure anything. Exit 2 is NOT a result.", file=sys.stderr)
    sys.exit(2)

def disassemble(path):
    import os
    if path.endswith(".dis"):
        try:
            return open(path, errors="replace").read().splitlines()
        except OSError as e:
            die(f"cannot read the disassembly {path!r}: {e}")
    base = os.environ.get("CAPSTONE_LLVM_BIN", "")
    if not base:
        die("CAPSTONE_LLVM_BIN is not set -- source capstone/tests/capstone-test-env.sh first. "
            "Without it the objdump path is '/llvm-objdump', which does not exist.")
    objdump = base + "/llvm-objdump"
    try:
        out = subprocess.run([objdump, "-d", "--triple=capstone64-unknown-elf", path],
                             capture_output=True, text=True, errors="replace")
    except OSError as e:
        die(f"cannot run {objdump!r}: {e}")
    if out.returncode != 0:
        die(f"disassembly of {path!r} failed (rc={out.returncode}): {out.stderr[:200]}")
    if not out.stdout.strip():
        die(f"disassembly of {path!r} produced NO OUTPUT -- nothing was scanned.")
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
    print(f"POSITIVE CONTROL (other region): {bad['other']} (MUST be >= 1, or this scan proves nothing)")
    print(f"the D3 trap site _handle_non_ecall: {control}  "
          f"({'STILL PRESENT -- the monitor fix is not in this firmware' if control else 'gone -- consistent with the D3 fix'})")
    for (r, f), n in per.most_common(15):
        print(f"   {r:9} {n:5}  {f}")
    if bad["other"] < 1:
        print("BLOCKED: the control did not fire. This scan measured nothing; its zero is not a result.")
        sys.exit(2)
    sys.exit(1 if bad["cap_text"] else 0)

try:
    main()
except SystemExit:
    raise
except BaseException as _e:                       # any crash is a CANNOT-CHECK, never a finding
    die(f"{type(_e).__name__}: {_e}")
