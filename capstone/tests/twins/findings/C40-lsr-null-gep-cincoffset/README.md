# C-40 — loop-strength-reduced pointer loops fault at -O1 and -O2: the exit test becomes a `cincoffset` off the null capability

**A COMPILER defect, found by the Tier 2a SLT twins on 2026-09-04.** The same SQLite
domain and the same three SQLLogicTest files AGREE with the native run at -O0 (select1:
1031 records, 0 failures) and fault at the first EXECUTED zero-base site at -O1 and at -O2, on the rebuilt
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
instruction; the -O1 image has seven, the -O2 image twenty (re-derived by the auditor from
the images that ran: domain_main 1/10, sqlite3VdbeExec 2/3, sqlite3WhereClauseClear 1/4,
fail 3/3).  All six failing runs trap at one of the two sqlite3WhereClauseClear sites
(-O1: pc 0x101cc9d4c, -O2: pc 0x101ccbb80; rd = x10, rs1 = x0), so most sites never
executed -- "first loop" would overstate what the logs show.  A `-disable-lsr` build of
the same module has zero such sites against eight in the control, which is the
no-rebuild experiment that settles whether C-40 is the only -O1 blocker.

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

## Open question: is C-40 present in the S-13 -O1 images?

Raised and then narrowed by the capstone session on 2026-09-04; recorded as an OPEN
QUESTION, not a finding.  S-13 (`tests/fpga-repros/S13-o1-dyn-rev-node-hang/`) is "at
-O1 the domain HANGS in the DYN/rev-node path with no exception": two distinct -O1 SQLite
images running a two-level join entered and never returned, board 2026-08-27, three days
after the c128 split.  Suggestive: C-40 is c128-era by construction (with i128 as the
carrier a GEP off null was integer arithmetic), its -O1 images die on QEMU after `G/enter`
before any result, and its sites sit in `sqlite3VdbeExec` (the interpreter loop every DYN
op is issued from) and `sqlite3WhereClauseClear` (the first statement with a WHERE clause).
Against: S-13's discriminator is a STALL -- aperture 225 = 0xd5 (dyn_wait_store_syncer +
dyn_wait_rev_res + stall_issue + mem_wait_flag) and store_syncer_req_set = 1 in all 8 S-13
boots and 0 in all 52 S-12 boots -- and a cause-24 fault in a domain without a trap vector
storms at address 0, which is fetch activity and gives no account of two syncer waits.  So
"an unhandled C-40 fault presenting as a hang" does NOT fit the measured signature; at most
C-40 is a second defect present in those images, a confound in the S-13 measurements.  (The
S-13 README never considers an unhandled fault at all, which is a gap on its own.)

Sites by function (page offsets in the SLT image): -O1: `domain_main` x1,
`sqlite3VdbeExec` x2, `sqlite3WhereClauseClear` x1, `fail` x3; -O2: `domain_main` x10,
`sqlite3VdbeExec` x3, `sqlite3WhereClauseClear` x4, `fail` x3; -O0: none.

The S-13 folder holds no images, hashes or build recipe, so the two images that wedged are
unreconstructable.  The test that remains: a FRESH -O1 two-level-join SQLite image from
today's tree, unfixed, on the rebuilt QEMU.  Cause 24 at one of the sites above before any
result means C-40 is present in that CLASS of image and is a confound for any -O1 board
measurement until fixed; no fault means the connection is dead.  Either outcome says
nothing about the two images that were measured.  Then, only if it faulted, the same image
with the C-40 fix and nothing else on the board -- one boot.
