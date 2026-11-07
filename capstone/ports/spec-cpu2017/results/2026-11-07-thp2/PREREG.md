# perlbench c3g and gcc train c3f on 2 MiB pages (kit bkit29-thp): pre-registration

Written and pushed before the board calls (7 Nov, 16:4x). The lead's focus: perlbench (1.444×) and c3f + THP.

**Kit.** bkit29-thp is c3f's stack with THP: the module is c3f's source (3fbed8df1fb8) plus the THP commits
(b23b38ce15b4, 37806776d76b), 5973baaff48a17db, param thp=1; the launcher is the quiet launcher with
CAPSTONE_VEXEC_HUGE_HEAP (959892e722a73f21: MADV_HUGEPAGE on heap arenas of 4 MiB or more); the board kernel has THP in
madvise mode (results/2026-11-06-thp). On mcf it removed 94 % of the fault exits and 343 s.

**Stages.** Copies of the two c3f/c3g stages that ran on bkit27-gtq, changed only by three readback lines: `@@THP`
(enabled mode, the module's thp) before the arm and `@@THP-VMSTAT` before and after (thp_fault_alloc, fallback,
split, collapse). Images, commands and parameters are byte-identical.

**Runs** (perf/b3/gt-run-thp.sh: flash C.5a, the list, flash bitstream 3 back), each after the fill run on its board
(results/2026-11-07-fill) ends with DONE:
- board 3: perlbench c3g-thp;
- board 2: gcc train c3f-thp, then perlbench c3g-thp (N = 2 for perlbench).

**References.** perlbench c3g (board 2): 33,667.7 s; inside 23,867.9 s; outside 9,799.8 s; 318 contexts, 282,236
steps; exits [quantum, service, fault] 19,264 / 265,988 / 84,937; rcyc 1,562 s, kcyc 3,263 s, ucyc 5,026 s.
gcc c3f (board 3): 4,253.3 s; outside 739.5 s; fault exits 2,875.

| # | Prediction |
|---|---|
| P0 | rc 0, specdiff PASS (perlbench 2/2 with test.out holding 0 FAILED; gcc 1/1), control passes, outputs equal the twin's; @@THP reads `[madvise]` thp=Y; thp_fault_alloc rises over the arm |
| P1 | perlbench fault exits ≤ 42,000 (half of 84,937) |
| P2 | perlbench run < 33,667.7 s; point 32,300-33,200 s: most of rcyc's 1,562 s goes, part comes back as 2 MiB zeroing on 318 short contexts |
| G1 | gcc fault exits ≤ 1,000; run ≤ 4,253.3 s; point 4,200-4,245 s (1.20× Plain is 4,218.6 s) |

**What would refute what.** P1 holding with P2 failing: the per-context cost of huge pages (zeroing, sweep) exceeds
the fault saving, and huge pages do not suit short contexts. P1 failing: perlbench's faults are not in 4 MiB+ arenas.

**Limits.** N = 2 for perlbench on two boards, N = 1 for gcc; cross-board against c3g (board 2) for board 3's run.

## Addendum 1 (7 Nov, 21:2x, before its board call): board 2's perlbench run returns, after perlbench -rg-nd there

Board 2's perlbench c3g-thp was withdrawn before any board call for perlbench -rg-nd (results/2026-11-07-nodiag). It is
queued again on board 2 behind that run (perf/thp4, the same stage and kit), for N = 2 with board 3's. Predictions
unchanged (P0-P2).

## Result, perlbench on board 3 (8 Nov 03:32): THP never engaged; P1 and P2 FAIL. Board 2's run is withdrawn

- 34,340.7 s (1.020× c3g's 33,667.7 s; 1.475× Plain's 23,277.1 s); inside 23,889.9 s (100.53 G), outside 10,450.7 s;
  rc 0, specdiff 2/2, test.out 317 ok / 0 FAILED, outputs equal the twin's. **P0 holds** except its last clause.
- **thp_fault_alloc stayed 0** over the arm (@@THP-VMSTAT before and after), with `[madvise] thp=Y`: no huge page was
  ever allocated. The launcher advises heap arenas of 4 MiB or more; perlbench's are smaller.
- **P1 FAILS:** fault exits 86,610 (c3g 84,937). **P2 FAILS:** slower, not faster. The +2.0 % is board 3 against board 2
  and the THP kernel together; not separated.
- Board 2's perlbench c3g-thp (Addendum 1) is withdrawn before its board call: it would read the same zero.

## Result, gcc on board 3 (8 Nov 06:24): G1 FAILS; THP formed 4 huge pages

- gcc train c3f-thp: 4,294.1 s (1.0096× c3f's 4,253.3 s on the same board), inside 3,514.6 s, outside 779.5 s;
  rc 0, specdiff 1/1, control passes. thp_fault_alloc 0 → 4 over the arm; fault exits 2,952 (c3f 2,875).
- **G1 FAILS** on every clause. gcc's heap arenas under the launcher's 4 MiB advice bound are not given huge pages,
  and the four it did get changed nothing.
- **Both programs read slower on the THP kit** (perlbench +2.0 %, gcc +1.0 %) without using huge pages: the kernel
  with THP and compaction costs about 1-2 % by itself. THP stays a lever for mcf only (its arrays are large arenas).
