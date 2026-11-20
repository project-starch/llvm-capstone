# Bitstream 1: P1 + the R-76 `ccsrrw` privilege term — pre-registration (2026-11-20)

Written before the full build is launched. Programme: `origin/spec-silicon:capstone/docs/plans/2026-11-20-post-submission-programme.md`
§7.1 item 1.

## What is built

capstone-ariane `p1-ccsrrw-priv` 9e801ab5e. Its tree differs from P1 (d83d3a633) in `core/csr_regfile.sv` only (+9 lines: the term
of 7180a83b6, byte-identical, and its comment, corrected to name the reachable supervised case) plus tests under `verif/`. Control tree for the arms: `p1-ccsrrw-base` f56263c24 = d83d3a633 + the same
tests; `git diff p1-ccsrrw-base 9e801ab5e` = `core/csr_regfile.sv` only.

The term: `ccsr_en && priv_lvl_o != M && !sup_active_q && !(DebugEn && debug_mode_q)` raises `privilege_violation`. Every input
is already an input of `privilege_violation`, which is on the standing UNOPTFLAT list: a loop-cone edit with no new signal.
Synthesis evidence for the same term is B5a's sealed full build `b5ar76-7180a83b6-full-out` (routed, 0 errors, 189,920 LUT
placed, WNS −13.615 ns), flashed on board 2 on 2026-11-20 with a clean Tier 1 (runtime lane). So no synthesis-only gate: the
full flow, under `synth-guard.sh`.

## Pre-synthesis gates (readings, not predictions)

All on capstone-ariane, S12_MEM_DELAY 12, `rtl_dirty=no` on both trees.

- **Directed arms**, two-sided (fix `.sup/b1arms` on 1c26498c1, whose core/ equals 9e801ab5e's but for the comment;
  control `.sup/b1base` on f56263c24). Every reading as pre-registered in the test headers:
  - u-ccsrrw and u-ccsrrw-scr: P1 0x8 0x4 A B (U reads and replaces M's capability: the hole, on P1 itself); fix 0x2 0x0 0x0 A.
  - u-ccsrrw-m (the M control): 0xb 0x4 A B on both.
  - u-ccsrrw-cih (`ccsrrw t1, cih, x0` from U, the runtime's probe form): P1 0x8 0x4 0x0 0x0; fix 0x2 0x0 0x0 0x0.
  - sup-ccsrrw-low (R-76's unexercised case: M arms a seal, mret to S, S calls it, the domain runs supervised at S) and
    its U, ECALL, ECALL-U and CPMP1 variants: identical on both trees. The stub's four ccsrrw on CSCRATCH and CEPC
    retire at privilege 1 (U: 0); csupstatus 1 after return; under ECALL csupstatus 5, csupcause 9 (U: 8), epc +0x10.
  - sup-ccsrrw-low-ctrl (S issues ccsrrw outside supervision): P1 0x9 0x4 (retired); fix 0x2 0x0.
  - sup-guards: identical on both trees (918,832 retired, the same readings).
- **Lint gate** (`verif/sim/rtl-lint-gate.sh`, `.sup/lint-b1` on 9e801ab5e and `.sup/lint-b1base` on f56263c24, same
  footing): PASS on both, every counter identical: LATCH 44, MULTIDRIVEN 3, ALWCOMBORDER 0, COMBDLY 0, UNOPTFLAT 39,
  BLKSEQ 2, UNDRIVEN 25, UNUSEDSIGNAL 1480, CASEOVERLAP 2, SELRANGE 100, ANVIL_UNOPTFLAT 0; the 39 UNOPTFLAT names
  identical (ccsr_en among them, as before the change).
- **95-test sweep** with the RVFI trace compare against P1 (`sim14-wt/.sup/pwsw6`, d83d3a633): run `b1-wt/.sup/swb1` (rtl 9e801ab5e, dirty=no,
  anvil prologue OK, SWEEP-RAN 95, 92 logs, every one newer than the run's PROVENANCE): values 92 identical / 0 differ / 3 missing
  on both sides (rand, perf_tcache_random, perf_riscv_random, as on every sweep); RVFI traces 92 identical / 0 differing
  (sweep-tools 48969d26a sweep-compare-rvfi.py); cycles 0 of 92 differ (sweep-compare-cycles.py). The cycle compare's control:
  the same tool on the I$-64B variant's sweep reads 67 of 92 differing.

## Full-build keys (P1 reference: `pw-d83d3a633-full-out/grade-full.txt`, read 2026-11-20)

| key | P1 | prediction | if missed |
|---|---|---|---|
| run peak memory | 27.39 GB | within 40 GB guard; mapping in P1's class | guard kill → read the partition the runaway worker mapped; no retry on identical inputs |
| placed LUT (top) | 187,368 | 187,368 ± 2,000 (the 2,000 top floor) | per-module census against P1's `utilization_hier.rpt` before any flash |
| placed FF | 76,543 | 76,543 ± 2,000 | as above |
| routed | 0 errors, 241,265/241,265 | 0 errors, 0 unrouted | ROUTING FAILURE BRANCH: placed area and congestion are recorded; **no WNS is a result**; not flashable; census of the congested window against P1's placed checkpoint; to runtime |
| placed congestion | level 4 | ≤ 5 | as routing |
| check_timing loops | 0 | 0 | worst-loop members named against P1's |
| WNS | −13.173 ns | inside the null band ±3.74 ns of P1: [−16.913, −9.433] | outside on either side → worst-20 path families against P1's before any flash; to runtime/the lead |

## Silicon keys (runtime lane runs the board)

- Tier 1 identical to P1's (the measured path holds 0 `ccsrrw` words).
- `ccsrrw` refused below M outside supervision: the runtime's U-mode probe `ccsrrw t0, CIH, x0` (0x001072db) under a SIGILL
  handler raises SIGILL (cause 2) on this bitstream, where P1 retires it with t0 = 0 (source reading: capstone_csr_unit.anvil's
  CIH case never returns the CSR to rd; csr_regfile's ccsrrw block is guarded by `!privilege_violation`).
- gcc, mcf and perlbench (the stages P1 was measured on) within 0.2 % of P1's cycles. Cost terms counted: the term adds no
  cycle on any path that retires no `ccsrrw` below M; the SPEC images hold 0 `ccsrrw` words (runtime lane's count); the clock
  is fixed. A difference above 0.2 % is therefore not a cost of the term and is attributed before it is reported.

Every reading on P1 or this bitstream carries "key 5 NOT MET".

## Hand-over

To the board lane for the full build (its pre-registration and dry run: llvm-capstone `board/meas-pw` 120d80c0e4e5), with the
runtime lane in copy. Flashes and board time stay with the runtime lane.
