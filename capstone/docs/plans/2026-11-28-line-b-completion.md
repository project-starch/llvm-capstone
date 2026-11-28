# Line B to completion: everything the paper needs, then SPEC train last

Status: approved by the lead, 2026-11-28. It extends the Line B campaign plan (docs/plans/2026-11-25-line-b-final-campaign.md;
its 28 Nov audit and decisions are on runtime/plan-audit-2026-11-26 8880ecd564ea) and the post-submission programme
(origin/spec-silicon 9228d775aac9). Owner: the runtime lane (coordination); per-item owners below.

## Context

The lead asked for a maximal audit and one plan that accounts for the old plans, every proposal, the paper and
everything else, with SPEC train left to the very end, when everything else is 100 % ready.

Three read-only audits ran on 28 Nov (the submitted paper ebc1700a against every Line B reading; every plan, proposal
and ISSUES entry; every lane branch and what is unlanded). Their load-bearing findings were re-checked at source:
the paper's perlbench is makerand + test.pl (spec-runtime.dat: Plain 10,806 s, Full 13,074.67 s); Line B excluded
test.pl on purpose (spectest Amendment 5) and a test.pl child faults with cause 28; Table 3 is stock against
protected, both Full, run time and "cycles outside the kernel"; LSRH and Zcb failed their board gates (e09ecb0844a4,
d0c5af96e033); the JTAG busy timeout has three instances (ISSUES:4989 on 24 Nov, plus 27 Nov PL3 and CS-x264b).

**The one-line state:** the Full and +Spatial seals are final and measured, the SPEC test campaign is two thirds read,
and four of the paper's results have no Line B backing at all: every "against Plain" number (Plain never booted),
the perlbench row on the paper's input, Table 3 (only CPython at N=1), and the embedded suites. None of the Line B
readings of 26-28 Nov has reached dev or the measurements doc.

## 1. Where Line B stands (each row verified at source)

| Area | Done | Open |
|---|---|---|
| Seals | Full b42c8c644 (WNS +0.605, 150,303 LUT), +Spatial 14f2ef81a, Plain rvbase 8db8f9cda (on silicon in P1-era train rows), stage-2 N 796af23d9 (WNS +1.796) | N has not run the qualifying kits |
| SPEC test campaign (b5b2-campaign) | Phases A, C, C2 and board 2's long rows: P1 on 42 boots, P2, P4 within 0.06 %; P3 on imagick and gcc | Phase B (boards 1 and 3, about a day); P3 on perl undecided; geomean and intervals |
| Ports (gcc, perl) | Full + ports N=3: gcc +5.05 % (v1) / +4.92 % (v2), perl +0.68 % (makerand) | on the paper's inputs (gcc train01, perl full test) |
| Stage 2 | GO on gcc only (−1.62 %), perl/imagick/nab flat, outputs equal; seal cost on perl not detected | x264 bracket on N (compiler, ~13 h, PREREG not yet pushed); decided out of this campaign (8880ecd564ea) |
| Placement control | perl 11.38 s, nab 2.21 s from relinking alone | the paper's "two runs within 0.1 %" does not hold for perl on Line B (wording, lead) |
| Record cache | gcc, perl, mcf LB 1024x8 NO BUILD (a7038b49974f, the tick log) | RC UNDECIDED, no owner of c's bound |
| Hardware cost | rebuilt from routed reports: Full 2.655x CVA6, +Spatial 2.457x (26801c60085d) | paper update (lead); no record-cache hardware |
| Section 5/6 claims | record TLB 16, genlim, relrace, cause 30 and growth, NOGROW, CSMINT N=1 (74cd5392a34e, 371f7706e0e5) | the UNVERIFIED list in step 2.6 |
| Plain arm | images relinked 28 Nov for the campaign's rules (FW_FAST_RDTIME, groups, bounds), twin replay all PASS, grader copy with both controls, PREREG 932491ce4360 | the board session (board 2) |
| Records | spectest and ticklog kits landed on dev (46ff1a089f63, 39ced56c3da2); measurements doc updated (ff27111710fa); I-17 (2fe83451f41b); campaign interim (cf82e5bc09b5) | before 28 Nov the measurements doc was untouched since 22 Nov; 52 runtime, 27 compiler, 99 board, 202 nested commits unlanded; the binding grader v2 exists only on board/meas-pw; state docs and the ISSUES header stale since 25 Nov / 16 Oct |

## 2. The paper's rows and claims without Line B backing (the work list)

1. **Every "against Plain" number** (Fig. 8, +7.5 % / +8.7 % geomeans, 20-22 % heavy, the 30 % bound, the timer
   appendix): needs the Plain arm on rvbase with the campaign's rules.
2. **perlbench on the paper's input** (makerand + test.pl, about 3.6 h per run): needs the cause-28 fault in a test.pl
   child explained first (spectest Amendment 5; the twin shows `.perl-native` cause=28 at pc 0x3ff7c01498 in every
   K-era boot), then a full-test stage on all three arms.
3. **gcc on train01** is the paper's gcc footing: it goes with train (section 4, phase 5).
4. **Table 3** (8 applications, stock against protected nested allocator, both Full, run time and cycles outside the
   kernel): only CPython (+6.8 %, N=1, 8 benchmarks against the paper's 7) and a memcached qualification exist.
   PostgreSQL and FFmpeg have no Line B port read; SQLite and mruby only on 904cd9c6b.
5. **Embedded suites** (BEEBS 93 / upstream 80 / malloc 13, RV8 7, CoreMark-derived): no Line B images.
6. **Revocation cost on silicon** (100,000 and 450,000 objects; interruptible CREVOKE changes what "latency" means).
7. **Security corpus** (159 cases, 9 applications; Juliet 366 pairs; "QEMU and FPGA give one set of verdicts"): on
   Line B only the SQLite and PHP engine repros, ungated, with a registered grader that failed on console splices.
8. **Design claims to settle by source or a small test:** the table's initial size (the paper's one batch against
   P1's four pages), CREVOKE "in one pass" (Line B re-reads), "fails on a stale capability before changing any
   record", the kernel's per-region CDERIVE and munmap CREVOKE, the record field layout, the lifetime-patch line
   counts for the one-operand revisions, the reuse-distance figure on one-operand builds, the "dedicated mode" wording.
9. **Wording the lead decides:** the 0.1 % repeat claim, the switcher and mixed-CSR cost rows (gone on Line B), the
   Plain macro naming d31345cf3 against 8db8f9cda, the record cache's "no hardware", "both a QEMU and an FPGA
   implementation".

## 3. Recommendations (what to do, what to drop)

- **Keep b42c8c644 / 14f2ef81a / 8db8f9cda as the final arms.** Stage 2 stays a separate result; after the x264
  bracket it is the lead's call whether it becomes a paper row.
- **Drop for this paper, with a written disposition:** RC hardware (no owner of c's bound; the LB evidence says the
  record path's levers are small), record-TLB LRU and pcc_hold (no model, no evidence), auipcgp (QEMU only, no RTL),
  C-63 (no workload needs it), the I-cache 64-byte variant (only if board time remains), LSRH and Zcb (failed their
  gates). Each gets one line in the Line B plan's dispositions.
- **Do, in this order of value:** Plain, perl full test, Table 3, embedded suites, the corpus, revocation cost, then
  train. Plain first because every overhead number in the paper is relative to it.
- **Fix the record debt now, before more results pile up:** the measurements doc, the landings, the grader on dev.

## 4. The order of work

Board time is the critical path; everything without a board runs beside it. Every board run is pre-registered and
pushed first, passes preflight-board-run.sh (exit line kept), has a stubbed dry run of its orchestrator, a known-good
control, and is graded at source by the registering lane with a positive control.

### Phase 0, now, no board (runtime unless named)
1. **Measurements doc:** enter every Line B reading of 26-28 Nov (campaign interim, spectest x264b, ticklog and the LB
   verdict, stage 2 with its seal control wording, placement, hardware cost, LSRH and Zcb failures, the Linux boots),
   each with its commit, image hash and N.
2. **Landing on dev, squashed per kit, after a `git log <base>..origin/dev` rebase check:** spectest and ticklog kits
   (final) now; campaign when Phase B ends; board lane: grade-spec-chain-v2.py and the graders the kits cite, the
   console-reconnect client, the DECLARED-checker fix, its board records; compiler: the stage-2 and spec8 kits;
   nested: its lane doc and RESULTS. Each lane lands its own.
3. **ISSUES:** the JTAG debug-module busy timeout as its own entry (three instances, two seals, first load after a
   flash in one, all recovered by one rerun); Line B dispositions for the CSRUNV-era entries (S-21, S-22, S-25, S-26,
   S-29, R-44, R-69); R-77's fix status and S-35's owed 28fcbb646 control stated plainly; the header date.
4. **State docs** (current-state, current-next-step) rewritten for Line B; the Line B plan's flagged inconsistencies
   I-1..I-8 corrected (perl port built, the mode's home, R-77/R-80 wording, LSRH/Zcb failed, item 12 done, the
   JTAG count, the programme location, the Line A levers' disposition).

### Phase 1, boards now to about +1 day
1. Board 1 (CS) and board 3 (CF): finish Phase B. Then the campaign RESULTS final (P3/P4 on long stages, Full /
   +Spatial ratios and geomean with ranges).
2. **Plain on board 2** (it stood idle and Plain's PREREG was pushed first, runtime/b5b2-linux 932491ce4360; the twin
   replay and both grader controls ran before it): flash rvbase, PLM-g3 as Tier 1 and size proof, short groups ×3,
   g1b ×2, x264b ×1, flash back. x264's encoded stream differs from B5TL1's by tens of bytes on the twin (not
   attributed; the compiler lane's discriminating test, a fixed --qp encode per arm, goes in phase 2).
3. **The compiler's x264 bracket on N** on the first board free after Phase B, once its PREREG and C10 declared dirs
   are pushed; then the flash back to that board's resident.

### Phase 2, preparation beside the boards (no board)
1. **perl full test (nested + compiler + runtime):** reproduce the test.pl child's cause 28 on the twin, attribute it
   (a protection finding, a port bug or a harness effect), fix or register it; then a full-test perl stage (makerand
   + test.pl, bound about 20,000 s) for B5TL1, SP and Plain-matched, twin-replayed.
2. **Table 3 (nested):** per application a stock and a protected-nested image on the trap route, the paper's
   workload (speedtest1, pyperformance 7, pgbench-derived, 300-frame decode, ao_render, perlbench test, mc-crusher
   128 MiB, a tenth of bench.php). "Cycles outside the kernel" from the tick-log kernel's accounting (apply-ticklog.py
   v3: cycles − kernel share over the process window), which needs no new instrument. PostgreSQL and FFmpeg ports on
   Line B are new work; if one cannot run on the trap route (fork, threads), that row is disposed in writing.
3. **Embedded suites (compiler + runtime):** BEEBS, RV8 and the CoreMark-derived benchmark as B5TL1, SP and Plain
   images; twin check against the P1 outputs; PREREG.
4. **Security corpus (nested):** the 159 cases as Line B images where a port exists, graded by a splice-safe reader
   (resplice, tested on a spliced capture first); Juliet's 366 pairs on the twin; QEMU against FPGA verdict equality
   as a registered check.
5. **Revocation cost (rtl + runtime):** a silicon kit for 100,000 and 450,000 records on b42c8c644, measuring both
   uninterrupted time and the interruptible restarts.
6. **Design claims (rtl):** each item of section 2.8 settled by a quoted source line or a directed test, written into
   the paper-claims doc.
7. **Train readiness (runtime):** item 6 (twin memory check of every train input at 962 MiB), item 8 (train stages,
   with an external initrd or a split payload for inputs over the 26 MB cap), item 9 (the Addendum 231/232/t202s2
   re-registrations), Plain train images, SQLite `--size 100`. Built and twin-checked here, booted only in phase 5.

### Phase 3, the final campaign on the boards (three runs per arm, drift-balanced, board-constant pairs, intervals)
- perl full test on CF, CS and Plain; embedded suites on all three; Table 3 pairs on CF; the corpus on CF; revocation
  cost; any Plain long rows still at N < 3; stage 2 as its own arm only if the lead asks.

### Phase 4, consolidation
- All RESULTS final and landed; the measurements doc complete; paper-numbers-checker run against it; claim-auditor on
  every number that will reach the paper; a proposal of paper changes to the lead (rows, figures, wording of section
  2.9). The paper changes only on the lead's word, through a PR branch; the abstract never.

### Phase 5, the very end: SPEC train
- Only when phases 0-4 are 100 % done and the lead agrees: SPEC train for all three arms (gcc train01 is the paper's
  gcc footing), SQLite `--size 100`, three runs per arm, then its RESULTS and the measurements doc.

## 5. Questions that are the lead's (none blocks phases 0-3)
- Landing the measured runtime (trap/eval with trap-alloc-lean, partly the collaborator's series) and the measured
  codegen default on dev.
- Stage 2 after the x264 bracket: a paper row or not.
- Paper wording (section 2.9), the anonymous repository, collaborator contact, C-60's threat model, R-71's boundary.

## 6. Verification
- Every board row: preflight exit line, readback before and after, the control first, graded at source, image hash
  cited, positive control for every reader (one perturbed hash must read `1 differ`).
- Every new reader or grader: a negative and a positive control before first use, and a spliced-capture test.
- Every landing: precommit-scan.sh by absolute path from the lane's own worktree, `git show` of the squashed commit
  read before the push, never chained.
- Every number for the paper: cited by image hash and commit in the measurements doc, checked by
  paper-numbers-checker, and adversarially by claim-auditor where it rests on N=1 or contradicts a prior finding.
