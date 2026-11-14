# Allocator-lean, and the cost of forming global capabilities (2026-11-14)

This is the compiler lane's record for lane branch alloc-lean.
- **Allocator:** a cheaper Capstone mallocng for the same policy (patches 0012, 0013).
- **Global-formation codegen:** sized, and its mechanism partly ruled out.
- **Retractions:** two, both recorded below.

All figures are native-twin user instructions. Outputs were compared byte for byte, and each arm's own validation is
quoted.

## Where perl's +13 % sits

Capstone FRCDKMS against riscv-base RVF3, full test, both board-faithful: +9.88 G.
- The allocator family is 42.6 % of it. malloc took 126 instructions per call against 68, and free with get_meta about
  228 against 140.
- The interpreter is 59.1 %, spread over 1,300 functions.
- The history note on lane node-table-pool has the method (14-11-2026_21-00-00).

## 0012 and 0013 (same policy, checked)

- **0012**, malloc in one call:
  - The reviewed __malloc_alloc body becomes an inline helper. check-mallocng-policy.py proves it equal token for
    token, and fires on a one-token mutant and on a tree without 0012.
  - build-sdk.sh's currency check now reverses the patch series as a whole.
- **0013**, get_meta always inlined, with sizeclass and maplen read once after the secret check:
  - The first version read the whole bitfield word ahead of the avail, freed and secret checks. A claim-auditor
    refuted its equivalence claim: a forged meta capability would fault differently.
  - That version was not committed, and its SDK, arm and stopped twins are VOID in perf/faultesc/RESULTS.
  - In the committed version the compiled order equals the pre-0013 order.
- **Gate:**
  - tests/native-task/free-reject, 11 cases: outcome, fault cause and faulting instruction, decoded from the
    binary.
  - Its double-live case reaches get_meta's freed-mask assert on the spatial configuration. It was added after the
    audit found that every killed case died in heap_access first.
  - The hosted contract on the native route.
  - Twins.
  - The malloc and nodes contract suites cannot run on this lineage.

| arm | perl full (vs board-faithful FRCDKMS) | gcc train01 |
|---|---|---|
| 0012 (FRCDKMSa1) | -352 M (-0.40 %) | -17.6 M (-0.14 %) |
| 0012 + 0013 (FRCDKMSa3) | -736 M (-0.83 %) | -37.0 M (-0.30 %) |
| riscv-base, the carried hunks (RVAL) | -189 M (-0.24 %) | -11.6 M (-0.11 %) |

- riscv-base carries 0013 and 0012's context pointer, by the same-software rule. 0012's split has no plain
  counterpart.
- Its control: the plain musl copy rebuilt without the hunks equals RVF3's libc in all 1,344 members.
- Twin ratios: perl 1.1261 to 1.1194, gcc 1.1288 to 1.1266.
- Both riscv-base deltas were larger than predicted.

0012's context pointer (M2) bought nothing: the compiler still re-forms the bounded global at each use.

## Global formation

A formation is `auipc` (with addi, or the folded lo), `li`/`add` for the end, `scc gp`, and optionally `shrink`.
Tool: perf/cclever/gform.py, matching 86 % of perl's scc-gp executions.

| | formations | redundant (beyond one per call per global per function) |
|---|---|---|
| perl full, FRCDKMS | 5.99 % of user | 1.34 % |
| perl full, GPR2 | 3.20 % | 1.19 % |
| gcc | 4.59 % | 0.39 % |

- After gp, the redundancy sits in Perl_yyparse and S_regmatch. The biggest single items are regmatch's switch jump
  table, re-formed per dispatch (0.37 G), and bison's tables.
- The backlog's A5 figure (0.19 %) is superseded. Its 9 Nov profile came from a failing test.pl run.

**gcc's loss under gp** (Addendum 120): gcc has 25 KB of small data, and gp reaches 4 KiB. With
`-global-merge-min-data-size=17` the small globals are un-merged, so they need more formations and more spills
(scc gp +30.6 M, auipc +27.6 M, sp loads/stores +33.4 M against GPC). Keeping relaxable small data unhoisted
(0.55 % of perl) would not help gcc, so it is parked with gp.

**PCC-relative tables are an ISA question.** On silicon `auipc` yields an untagged integer, and a load through it
faults (cause 24). No instruction reads data through PCC (rtl-oracle on afd35e449). It is in the handover's "waiting
on the lead" list.

**Why the loops re-form. Mechanism NOT found.** Ruled out, each measured on perly.c:
- register-allocator rematerialization: 84 formations before and after allocation (RETRACTED; I had claimed it);
- the speculation guard alone: with `-avoid-speculation=false` the assembly is byte-identical;
- machine sinking alone.
The recorded next experiment: the tied-operand SHRINK, re-applied per use on a movc copy of a hoisted scc result.

## F2 (the last-mapping cache)

F2 is Capstone-only: plain musl's free has no table walk. The writers of the page-owner table are __heap_mapping's
two paths and __heap_unmapping's two paths, plus table creation in leaf_for, all under index_lock. Lookups take no
lock, so a cache needs a generation counter and a seqlock read: about 10 instructions saved per free, about 0.18 G on
perl. The alternative proposed is a 2-level table with the same writers (about 0.1 G).

**Measured: the 2-level table, not adopted (2026-11-15).**
- The change: a 2^18-entry directory of 2 MiB leaves, mapped at the first heap mapping. The diff is kept on the build
  host (perf/alean/f2-two-level-table.diff); it is not committed.
- The gate passed:
  - free-reject MATCH on 11 cases;
  - its mutant (skip the owner clear at the reuse put) MISMATCHes through the reissue trap;
  - the hosted contract is identical;
  - a claim-auditor did not refute it (no missing reader or writer across the repository and its submodules).
- But it loses on cost. perl full against 0012+0013:
  - user -110.7 M;
  - kernel +133.4 M;
  - pages +2,278;
  - one more service exit per exec (k3 +318), the directory's own mmap.
- gcc: user -6.0 M, kernel +3.3 M, pages +481 (its stage's arena_pin pins the directory).
- The lookup keeps three dependent loads; it saves only the mid index's arithmetic.
- The seqlock cache waits on the FRCDKMSa3 board pair.
