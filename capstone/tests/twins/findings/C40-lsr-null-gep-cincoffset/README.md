# C-40 — loop-strength-reduced pointer loops fault at -O1 and -O2: the exit test becomes a `cincoffset` off the null capability

**A COMPILER defect, found by the Tier 2a SLT twins on 2026-09-04.** The same SQLite
domain and the same three SQLLogicTest files AGREE with the native run at -O0 (select1:
1031 records, 0 failures) and fault at the first loop at -O1 and at -O2, on the rebuilt
QEMU (5dc356547d7f, 2026-09-04 22:34) with the branch compiler (no codegen change from
db079043).

## The fault

    capstone-qemu: cincoffset with an UNTAGGED rs1 -- pc=0x101cc9d4c rd=x10 rs1=x0 val=0x0 priv=3
    [CAPSTONE] domain halted by capability fault: cause = 24, pc = 0x101cc9d4c, tval = 0x0

`fault-lines.txt` is the console; `disassembly-at-pc.txt` shows the instruction at page
offset 0xd4c of the -O1 image:

    sqlite3WhereClauseClear+0xbc:   cincoffset a0, zero, s4
                                    beqz       a0, ...
                                    cincoffsetimm s3, s3, 0x50
                                    addi       s4, s4, -0x50

The loop walks `WhereTerm a[]` (80 bytes each); `s4` counts the remaining distance to
`aLast`, and the exit test `a == aLast` has become `(null + s4) == null`.  `cincoffset`
with rs1 = `zero` -- the null capability, untagged -- raises UNEXPECTED_OPERAND (24) on
the first execution, on QEMU and, by the spec, on the RTL.  The -O0 image has no such
instruction; the -O1 image has five, the -O2 image eight.

## The pass

clang's -O1 IR for the function is clean (a pointer PHI compared against `%arrayidx`).
The rewrite happens in llc: `after-loop-reduce.ll` is the dump after Loop Strength
Reduction, which reads

    %scevgep = getelementptr i8, ptr addrspace(200) null, i64 %lsr.iv
    %cmp19   = icmp eq ptr addrspace(200) %scevgep, null

SCEVExpander expresses a pointer-typed induction expression in a NON-INTEGRAL address
space as a GEP off `null` (it may not use inttoptr there), and the backend lowers a GEP
on `null` to `CIncOffset` with the null register as base.  Any loop whose exit compares
a pointer against an end pointer is a candidate, so this is almost certainly the -O1
blocker behind C-3 and the reason SQLite has only ever shipped at -O0.

## Fix (target-side)

`null + x` can never be a usable capability, so the backend should lower a null-based
`CIncOffset` the way it lowers `inttoptr`: an untagged value carrying `x` as its cursor
(`INSERT_SUBREG sub_cap_addr` into an undefined c128, which copyPhysReg emits as one
ADDI).  Comparisons then read the cursor, and a dereference of such a pointer faults at
the load, where it belongs, not at the arithmetic.  The immediate form
(`cincoffsetimm rd, zero, imm`, the shape `select-cap.ll` already guards) gets the same
treatment.  Pinned red-first as `llvm/test/CodeGen/Capstone/c40-null-base-cincoffset.ll`.

## Status

Root-caused; fix designed; the rebuild waits for the running Tier 2a driver so the
compiler under test does not change mid-run.  Registry ID C-40 (free in ISSUES.md and in
`git log --all` on 2026-09-04).
