# SPEC train on silicon, Capstone and Plain-with-cache, for the abstract deadline: pre-registration

Written and pushed before the first board call (8 Nov, 19:4x). The external collaborator asked for definitive train
numbers by Wednesday 20:00. Every board stops test-size work: the queued test runs (GDB-reader validation, nab/xz
flag pairs, gcc repeats) were cancelled at 19:2x. The two test runs still on the boards finish first: perlbench
udcnofp on board 3, about 22:30, and Plain-reuse x264 on board 2, about 23:50.

## What runs

C.5a (caplifive_vp2c_1fc2cd5aa) on all three boards. N = 1 per arm. For each benchmark, Capstone runs the best
configuration that already passed on the board at test size, so no configuration runs here for the first time.
Plain runs Plain-with-cache (musl 1.2.5 mallocng with the same mapping cache), the fair denominator of
results/2026-11-07-plainreuse.

| Benchmark | Capstone exe | Capstone kit (driver) | Plain exe (kit bkit27-gtq) | Train subset |
|---|---|---|---|---|
| perlbench | unify-nd 89398ba2be3127ab, or udcnofp 5996ba25b57a303c (rule below) | bkit30-unify (perf/unify/un-run.sh) | 3bbd77760d40d34a | all 5 commands |
| gcc | dcnofp 736f881f4958ac3b | bkit32-q32 (perf/exhpm/gt-run-q32.sh) | 4d9ddde9aee9b35f | train01 only: 200.c needs 1.62 GiB and scilab.c 1.24 GiB, against about 963 MiB of Linux memory (results/2026-11-05-gcctrain) |
| mcf | devmk 3633be7e2f887047 | bkit29-thp (perf/thp/gt-run.sh) | 4857348eb6938346 | all |
| lbm | devmk f0269eea1b555bd9 | bkit32-q32 | ac528d31b91e1eff | all |
| x264 | c3f a52c991174079595 | bkit32-q32 | 2d8170fd12b5114f | all |
| imagick | devmk f50dc465183e4f1f (+ imagevalidate 8ecf59b59b9bc166) | bkit32-q32 | a0b1046436f94696 (+ 52b5c95397740354) | all |
| nab | devmk 092ff45c0c519283 | bkit32-q32 | 7db6f9fe0f3c116b | all |
| xz | devmk 0d8878ad8a5046df | bkit32-q32 | 27d9c10dc11946b6 | command 0 only: the full stage's initramfs does not fit the 26,000,000-byte firmware cap (results/2026-10-28-board-test/hfl/bake-size-check.txt) |

- **Stages**: $CAPSTONE_WORK/spec/stages-train-fair. Each is a dereferenced copy (`cp -rL`, no symlinks, link
  count 1) of the existing train stage (Capstone: spec/stages-train-hfl, Plain: spec/stages-train-plain; gcc: the
  train01 stages already run), with only the executables replaced. Each copy's source exe was re-hashed after the
  copy. perlbench's Capstone stages carry the one-line `unset CAPSTONE_EXEC_DIAGNOSTICS` of the unify-nd test stage.
  The @@ markers of every run.sh equal the current test stages'.
- **Kits**: a configuration for each benchmark is a choice, not a single Capstone setting, and the table above is
  what is reported. gcc, lbm, x264, imagick, nab and xz share bkit32-q32 (the 32 M supervisor quantum gcc measured
  best). perlbench keeps the unify stack, which beat c3g on perl (results/2026-11-07-nodiag). mcf keeps the THP kit.
  Plain did not gain from THP (9,150.5 s with THP against 9,077.4 s without), so Plain runs without it.
- **perlbench rule**: udcnofp replaces unify-nd only if its test run on board 3 (results/2026-11-08-dcnofp Addendum
  2) ends rc 0 on both commands with specdiff 2/2 PASS. The board-3 script checks that line in
  out-udc-b3/chain-b3.log before it chooses the list.

## Schedule (estimates: test board time × native train/test instruction ratio, tens of percent)

| Board | Order | Estimate |
|---|---|---|
| 1 (free about 19:55) | Plain: perlbench 13.5 h, nab 14.5 h, mcf 12.7 h, imagick ~12 h, gcc 0.9 h | ~54 h, Tue ~02:00 |
| 3 (free about 22:30) | Capstone perlbench 18.6 h (unify); Capstone nab 16.7 h, xz 5.7 h, gcc 1.1 h (q32); Plain lbm 7.8 h, xz 4.8 h | ~55 h, Wed ~05:00 |
| 2 (free about 23:50) | Capstone mcf 15.7 h (THP); Capstone imagick ~14 h, x264 10.2 h, lbm 7.5 h (q32); Plain x264 11.3 h | ~59 h, Wed ~11:00 |

Longest and riskiest first on each board. perlbench train lost runs to OOM in October, and two of its five commands
ended rc 1 in the out-hfl run.

## Predictions and gates

| # | Prediction |
|---|---|
| T0 | every command rc 0 and specdiff PASS; every boot's control passes |
| T1 | each run within ±35 % of its estimate above |
| T2 | each ratio within ±0.10 of its test-size ratio (perlbench 1.38 or its udcnofp reading, gcc 1.20, mcf 1.24, lbm 0.96, x264 test pending, imagick 1.16, nab 1.15, xz 1.17) |

- A run that fails T0 is not reported as a train number. That benchmark falls back to its test-size number, marked
  as such. If the board has time before the deadline, the run repeats once.
- A ratio outside T2 is checked before it is reported: kernel time, peak pages and outputs against the twin.

## Addendum 1 (8 Nov 20:4x, board 1 running Plain perlbench): the stages' plan.json executable hashes rewritten

Every stage in spec/stages-train-fair kept its SOURCE stage's `executables` sha256 in plan.json after its exe was
replaced, for example gcc 5642377943d724c6 for the file 736f881f4958ac3b. The compiler lane found this on the
Capstone stages, and the Plain stages had it too. The files are the pre-registered images: every file hash equals
this record's table. run-stage.py never reads those fields (only make-stage.py writes them), so no run is affected. All
17 plan.json files were rewritten to their files' hashes (written to a new file and renamed; link count 1). Results
cite images by file hash.

## Addendum 2 (8 Nov 20:4x, before board 3's perlbench call): two perlbench train commands crash on the QEMU twin

The compiler lane's train twins (kit-unq, dereferenced copies of these stages, exe hashes re-checked) of BOTH frozen
Capstone perlbench images end commands 2 and 3 (scrabbl, splitmail) with rc 139 "Segmentation fault", specdiff 4/6.
Commands 0, 1 and 4 pass. Per command, unify-nd 89398ba2be3127ab gives rc 0/0/139/139/0 at 108.94/20.59/0.59/2.95/2.32 G
instructions, and udcnofp 5996ba25b57a303c gives the same rc pattern. Prior art: the 28 Oct board test (results/2026-10-28-
board-test, PREREG 376-383 and 1052-1061) ended the same two commands rc 1 with perl's "Out of memory!" at the
lifetime-node table's capacity (node_capacity 262,144, node_growths 0). The cause of the SEGV is UNRESOLVED; a
diagnostic rerun is on QEMU.

- **The schedule stays.** The crashes come early (0.6 and 3.0 G of ~135 G instructions), so board 3's run costs about
  what the three passing commands cost. Plain perlbench (board 1, running) records every command's time.
- **What is reported for perlbench train**: if commands 2 and 3 fail on the board too, perlbench is reported on the
  subset {0, 1, 4} (diffmail, perfect, suns; the 3 commands that pass), Capstone against Plain on the same commands,
  and marked as a subset, as gcc's train01 is. If a fix lands before Wednesday, commands 2 and 3 run again as their
  own short stage on the first free board.
- T0 is read per command for perlbench: commands 0, 1 and 4 must be rc 0 with specdiff PASS.

## Addendum 3 (8 Nov 21:4x): gcc is not re-run; schedule end corrected

- gcc train01 already has its numbers in exactly this configuration: Capstone dcnofp 3,924.9 s (results/2026-11-08-dcnofp
  Result, exe 736f881f4958ac3b, kit bkit32-q32, board 3) and Plain-with-cache 3,280.9 s (results/2026-11-07-plainreuse,
  exe 4d9ddde9aee9b35f). Both gcc lines are removed. Board 3's q32 list was rewritten before that driver opened it.
  On board 1 (chain running), an existing output directory makes the chain skip the stage (chain-board.sh:26).
- A move of Plain x264 to board 1 was tried and reverted within minutes. It would have ended board 1 later than
  board 2. The appended list line is neutralised the same way (an existing output directory).
- The schedule table's end times were miscounted by a day for board 1. Corrected: board 1 Wed ~01:00, board 3
  Wed ~04:00, board 2 Wed ~10:30 (SG time, which is the deadline's time zone). The last train number is due about
  Wednesday 10:30.

## Addendum 4 (9 Nov 00:4x, before any of these board calls): the final board plan. Train complete by Wednesday 20:00 SG

The lead's clarification: all of SPEC train must be done by the deadline (a hard condition). Plain runs first, since
the bitstream does not matter to it. The two benchmarks worst for Capstone (perlbench 1.33×, mcf 1.24× at test size)
run on the NEW bitstream E5 (stage-B + 20-bit lifetime index, so perlbench's scrabbl/splitmail can complete), and the
rest of Capstone runs on C.5a now. The optimisation quick tests queued for tonight (kfast on board 2, NO_HZ perlbench
on board 3) are cancelled. The running NO_HZ gcc pair on board 3 finishes first.

| Board | Order (estimates) | Ends |
|---|---|---|
| 1, gtq | Plain perlbench (running), nab, mcf, imagick, x264 | ~Wed 12:00 |
| 2, q32 then gtq | Capstone nab, imagick (C.5a, this record's stages and exes); Plain lbm; Capstone mcf on E5 (THP kit variant, after E5's gate) | ~Wed 09:00 |
| 3, q32 then gtq | Capstone x264, lbm, xz (C.5a); Plain xz; Capstone perlbench on E5 (unify kit variant, after E5's gate) | ~Wed 02:00 |

- Plain lbm and xz moved from board 1 to boards 2 and 3 (an existing output directory makes board 1 skip them).
- The E5 stages get their own addendum with the bitstream's seal, gate result, kits and exes before they run. If E5
  is not sealed and gated in time, perlbench and mcf run on C.5a instead: perlbench is then reported on its commands
  {0, 1, 4} (Addendum 2).
- The C.5a Capstone stages, exes, kits and gates T0-T2 are this record's, unchanged.

## Addendum 5 (9 Nov 05:0x, before any e5-native board call): perlbench and mcf train on the native bitstream

The lead's "new bitstream" for the worst benchmarks is e5-native: the native Linux-task integration
(native/linux-task-integration, ported to C.5a's model) on caplifive_e5native_14b0a57df.bit (sha256 b6698b7c…a475;
E5 + CSRUNV detach/adopt + the R-69 clamp). The rtl lane graded it 4/4 against E0, the FF delta is attributed by name,
32/32 vectors are pre-registered (land/rtl-evariants 5cc4e683a354), the CSCHECK firmware emulation executed in RTL
sim with every row as pre-registered, and leffe's E5 control was read. Native route on the board:
- kernel K-min (the stock 6.4.14 .config + RISCV_CAPSTONE_TASK only, plus the adapter's command-line parameters);
- firmware capstone-sbi native/cscheck-emu e04ecb8 with CSCHECK emulated;
- kits t3-smoke/bkit38-native-perl and bkit38-native-mcf; driver perf/native-train/nt-run.sh;
- table 2^19 records for perlbench, 2^18 for mcf.
The native agent's pre-registration is llvm-capstone native/c5a-board 8cc014c0bdd2. QEMU twins: mcf train at 936M rc 0
(peak 221,799 pages); perlbench train five commands rc 0, with a rerun for the full outputs.

Order, per board, with a wait for a free board before each step:
- **Board 3**, after its running Capstone x264 (C.5a): (1) the 32-vector gate (perf/e5ngate/e5n-gate.sh 3; a
  failure flashes C.5a back); (2) only on GATE PASS, nt-run.sh: a native SMOKE boot, then perlbench train (the train
  boot runs only on a smoke PASS); (3) if (2) does not complete, perlbench train on C.5a (bkit30-unify, udcnofp stage,
  reported on commands {0, 1, 4} as in Addendum 2); (4) Capstone lbm and xz on C.5a (q32), deferred from this board's
  list; (5) Plain xz.
- **Board 2**, after its running Capstone nab (C.5a): (1) the gate on board 2; (2) native smoke, then mcf train;
  (3) if (2) does not complete, mcf train on C.5a on the 4K q32 kit (stage 505.mcf_r-train-capstone-lifetimes-devmk,
  the same frozen exe 3633be7e2f887047). Not THP: the THP twins were killed at 80 G at 936M and 957M (compiler lane);
  (4) Capstone imagick (C.5a, q32), deferred; (5) Plain lbm.
- Reported per benchmark: the native-route time, or the C.5a fallback's, whichever completed. The native ratio's Plain
  is this record's Plain on the stock kernel: K-min differs from stock by RISCV_CAPSTONE_TASK only.

## Addendum 6 (9 Nov 07:3x, before any of it runs): board 3's perlbench fallback is E5c, so perlbench can complete in full

E5c = caplifive_e5c_246e7abc2.bit (sha256 ad6db31b…fa27): E5 (stage-B + 20-bit lifetime index) + the R-69 resume clamp,
csr_regfile only. Sealed by the board lane, full build 4/4 against E5 (WNS −12.723). Its vector gate is
perf/e5cgate/e5c-gate.sh (rtl lane, land/rtl-evariants c84cc235ec46), with vectors equal to e5-native's and E0's.
If board 3's native route does not complete (Addendum 5, step 3), the fallback is now:
(a) the E5c gate; (b) on PASS, perlbench train udcnofp (5996ba25b57a303c) on kit t3-smoke/bkit39-e5c-unify. That kit
is bkit30-unify with the E5c bitstream and the 20-bit unify module 6f4cea6f3d99c9c1, built from the same source and
recipe as 13a5ad66 with only CAPSTONE_VM_INDEX_BITS=20; its .sdata reads index_bits 20 and 1,048,576 records. All
five commands are expected to complete (the uncapped twins: scrabbl 414,817, splitmail 410,594 live nodes). Driver
perf/train/un-run-e5c.sh. (c) If the E5c gate fails (the board is back on C.5a), perlbench on C.5a with commands
{0, 1, 4} as before. The deferred stages follow unchanged. Board 2's mcf fallback (C.5a, 4K q32) is unchanged: mcf
peaks at 14 k live nodes.

## Addendum 7 (9 Nov 10:0x, before it runs): Capstone xz train runs on its best test configuration, kit bkit25-nset

The rule of this record is the best configuration that passed on the board at test size. For xz that is kit
bkit25-nset (retire_exit_sweep=2, 8 M quantum, module 645749816c02751e), not bkit32-q32. Its run (out-ns-b2x, board 2,
C.5a) read 8,084.2 s, specdiff 12/12 PASS, outputs equal to the twin's: 1.005× Plain-with-cache's 8,044.1 s. It ran
the same 50.1 G instructions as out-all-b1 (9,398.2 s), so the difference is outside time. q32 never ran xz at test
size. T2's xz reference changes from 1.17 to 1.005, band ±0.10.
- Board 3: xz leaves perf/train/b3d-q32/b3.list (lbm stays there). Once nat-board2.sh (pid 3671072) ends, a waiter
  (perf/train/train-board3.sh) runs the same stage, 557.xz_r-train-capstone-lifetimes-link-c0-devmk (exe
  0d8878ad8a5046df, command 0), through perf/train/gt-run-nset.sh: the q32 driver with the kit, its module/vexec
  hashes, the 8 M quantum, the s26 wrapper and the sweep-2 params line checked instead. Output out-train3-b3-nset.
- Plain xz is unchanged and runs before it, as step 5 of Addendum 5.
- Cost: none on board time; xz moves after Plain xz on the same board, about 5.7 h, ending about Wed 10:00.

## Addendum 8 (9 Nov 12:4x, before any of it runs): Plain first on all three boards, as the lead decided

The lead's order was Plain first, then the worst benchmarks on the new bitstream. Only board 1 ran Plain; boards 2
and 3 ran Capstone C.5a stages first, which put all of Plain in one queue ending Wednesday. The runs in progress are
not stopped (board 1 Plain nab, board 2 Capstone nab, board 3 Capstone x264). Only waiting wrappers are replaced; none
had started a board step. Stages, exes, kits and drivers are unchanged; only their board and order move.
- **Board 1:** Plain nab (running), Plain mcf; imagick and x264 leave its list (SKIPPED out dirs); then Capstone
  imagick (q32) and Capstone xz (nset, Addendum 7).
- **Board 2:** Capstone nab (running); Plain imagick, Plain xz; then Addendum 5's native mcf with its 4K q32
  fallback; then Capstone lbm (q32).
- **Board 3:** Capstone x264 (running); Plain x264, Plain lbm; then Addendum 5/6's native perlbench with the E5c and
  C.5a fallbacks. Its deferred q32 and Plain steps are now empty.
- Estimates: Plain complete about Tue 13:00; native perlbench and mcf end about Wed 04:00-05:00; all about Wed 11:00.
- Cost, accepted: a native run that fails late leaves no time for its C.5a fallback before the deadline. That
  benchmark then reports its test-size number, marked as such (rule T0).

## Addendum 9 (9 Nov 13:4x, before it runs): xz train command 1 (IMG_2560.cr2.xz) on both arms, so xz runs its full train input

xz ran command 0 only because the full stage bakes to 42,633,736 B. The 26,000,000-byte cap is this project's own
kit setting (BOARD_FW_CAP, preflight C10's declared-bake value), not a board limit. A board campaign plan records
the console's upload limit as 32 MB (HTTP 413). Command 1 alone was estimated at 34,245,128 B
(results/2026-10-28-board-test, "above every proven upload"), so this is also the first test of that limit.
- **Stages** (spec/stages-train-fair, made by perf/train/make-xz-c1.py from each arm's c0 stage, the same selection
  make-stage.py --commands makes; every substitution counted and asserted): 557.xz_r-train-capstone-lifetimes-link-c1-devmk
  (exe 0d8878ad8a5046df) and 557.xz_r-train-plain-rv64-link-c1r (exe 27d9c10dc11946b6). Input IMG_2560.cr2.xz
  (sha256 6e17c28f…). One command, one host specdiff. Apart from the argv and file names, run.sh is the c0 stage's.
- **Kits:** bkit25-nset-fw36 and bkit27-gtq-fw36 are copies of bkit25-nset and bkit27-gtq with BOARD_FW_CAP 36,000,000
  and fresh SHA256SUMS. The originals are untouched. Drivers gt-run-nset-fw36.sh and gt-run-ga2-fw36.sh check the
  copies.
- **QEMU twin** (census-train, lifetimes, 936M): command 1 runs 50.97 G instructions (command 0: 112.65 G), peak
  142,631 pages. Estimates: Capstone ~2.6 h, Plain ~2.2 h.
- **Order:** Plain c1 on board 2 after its Plain xz c0, before the native mcf; Capstone c1 on board 3 after native
  perlbench. Both deadlines keep the end before Wed ~14:00.
- **Predictions:** X0: rc 0 and specdiff PASS on both arms. X1: firmware 33-35 MB. X2: if the upload is refused (HTTP
  413), the stage reads NO DATA. Each arm costs one boot, and xz reports command 0 alone, as now. X3: xz's full train
  time is the sum of the two commands' run seconds per arm, and its ratio lies within ±0.10 of command 0's.

## Addendum 10 (9 Nov 14:4x, before any of it exists on a board): option B (ids55) for the Temporal train, with decision rules

The lead and the external collaborator chose option B (rtl-ids55, 325966ee4: a 55-bit lifetime id, 30-bit generation +
25-bit slot) over option A (E5's 30-bit id re-split to 10 + 20). B's software side exists (the flat ids55 module, its
QEMU twin, the 30 Oct board test on bitstream 4). B is not on C.5a's lineage, so three bitstream variants are being
prepared in parallel. None is used until it is sealed and passes its board gate.
- **V1 e5n-ids55:** e5-native + B (capstone-ariane b91439bf7 and successors). For native perlbench and mcf.
- **V2 e5c-ids55:** stage-B + the R-69 clamp + B, without DETACH/ADOPT. For the module route; the hedge if V1 does not
  route.
- **V3 c5a-ids55:** C.5a + B only. The smallest delta from the bitstream the current numbers ran on.
- **Software:** native-route ids55 kernel/firmware (native lane), and module-route ids55 kits (bkit41-ids55-*). Each
  gets QEMU twin gates and train twins at 936M first.

**Decision rules** (the order of preference; every switch is its own addendum with hashes before it runs):
- D1, perlbench (board 3) and mcf (board 2), at their start (about Tue 10:00):
  (a) native on V1 if V1 is sealed, its board gate passes, and the native ids55 kits passed their twins;
  (b) else the module route on V2, else on V3, if sealed, gated, and the module ids55 kits passed twins;
  (c) else Addendums 5/6 unchanged (option A, e5-native native route, C.5a and E5c fallbacks).
- D2, the Capstone runs not yet started (imagick on board 1, lbm on board 2, xz commands 0 and 1): on the best sealed,
  gated B variant whose q32/nset ids55 kit passed its twin; else C.5a as now.
- D3, the Capstone runs already done on C.5a (gcc train01, nab, x264): reported on C.5a, marked so. They rerun on B in any
  board slack before the deadline, longest gap to the target first, and otherwise after it.
- Plain is unaffected.
- Every benchmark's row names its bitstream. A row on option A or C.5a is marked as such, since the paper describes B.
- A gate failure on a B variant flashes C.5a back and falls to the next rule; no stage runs on an ungated bitstream.

## Addendum 11 (9 Nov 16:1x, before any B board run): unfixed B variants only where the hazard census reads zero

The rtl lane found a defect in option B's load rule (R-70, to be filed). A plain store to doubleword 1 of a tagged
granule, still in the store buffer, does not order a following plain load of doubleword 0. That load can return
the projected cursor instead of the raw word, depending on timing. QEMU always returns the raw word. The fix (V1f
0ef470559, V2f c95525476, V3f 596c36fc7: the D-cache read controller treats a granule with a pending plain store as
untagged) read as pre-registered in RTL simulation on V1f. All six variants are in synthesis.
The compiler lane's QEMU census counts that pattern (exact distance K, tag read before the store): gcc test reads 27
tagged events at K ≤ 16. Its positive control reads exactly.
- **Amends D1-D2:** a benchmark runs on an UNFIXED variant (V1, V2, V3) only if the census reads zero tagged events at
  K ≤ 64 for that benchmark's exact exe at that size. Otherwise only a fixed variant (V1f, V2f, V3f) is eligible.
  With no fixed variant sealed and gated, the benchmark falls through to option A or C.5a as in Addendum 10.
- The native route's firmware needs the fence (capstone-sbi 4f51541) on unfixed variants only. It is correct on both.

## Addendum 12 (9 Nov 17:0x, before it runs): quick test-size runs tonight decide native vs the kernel module

The collaborator asked for a decision tonight between the native Linux-task integration and the kernel-module route,
from quick test-size board runs, before the Capstone train is committed. Running stages are not stopped. Board 2,
after its Capstone nab (about 18:00), runs one session on e5-native (caplifive_e5native_14b0a57df, option A, sealed):
1. its 32-image gate (perf/e5ngate/e5n-gate.sh 2);
2. native: smoke, then imagick, nab and lbm at test size (native lane's qt stages, the same lifetimes exes as below);
3. module route on the same bitstream: kit bkit39-e5n-unify (bkit39-e5c-unify with only BOARD_BITSTREAM changed,
   fresh SHA256SUMS; driver perf/train/un-run-e5n.sh), stages stages-eval-b1/{538.imagick_r,544.nab_r,519.lbm_r}-test-devmk
   (lifetimes exes 8ecf59b59b9bc166, 092ff45c0c519283, f0269eea1b555bd9).
Then board 2's chain continues.
- **Reading:** per benchmark, native / module run time on the same board and bitstream, plus each one's ratio to Plain
  with the cache (imagick 19.46 s, nab 862.99 s, lbm 1,895.0 s).
- **Prediction:** native ≤ module on every benchmark. The gain is largest where exits dominate: nab and imagick's
  outside time. lbm is near 1.00.
- **Decision rule:** native takes the remaining Capstone train if it is faster or equal on at least 2 of the 3, and
  slower on none by more than 2 %. Otherwise the module route does. Plain then keeps one board, and the chosen route gets
  two, by a further addendum that also covers the queue change.

## Addendum 13 (9 Nov 17:3x, before it runs): the same quick tests on V1f (option B + the R-70 fix), board 2, after Addendum 12

V1f = capstone-ariane 0ef470559 (e5-native + option B + the R-70 load-rule fix) routed and sealed 17:22:
caplifive_e5nids55f_0ef470559.bit, sha256 06f6be7e3cf977908d0f103526929d97643c8241e30be6df3133d653867d86bf. Full grade
4/4 vs e5-native: WNS −13.873, 0 loops, routed clean; placed 179,757 LUT, placer congestion 5 (at the key's limit).
After Addendum 12's session, board 2 runs perf/train/qt-board2-v1f.sh. It shares that session's structure.
1. V1f's 32-image gate (perf/v1fgate/v1f-gate.sh 2; vectors from the rtl lane's V1f simulation; C.5a flashed back on a
   FAIL). No stage runs unless it passes.
2. native: the same four test programs (gcc, imagick, nab, lbm), through the native lane's ids55 kits (firmware
   capstone-sbi 4f51541, its RTL arm passed on V1's model);
3. module: the same four module stages through bkit41-ids55-unify (m55-unify 63d6081632249045, 2^20 table) and
   perf/ids55mod/run-ids55-unify.sh.
- **Reading:** per benchmark and route, V1f against e5-native (B against A) on the same board, the same exes and the same
  route.
- **Prediction:** in-context cycles equal within 0.5 % for B and A on every benchmark (the rtl lane predicts a cycle
  delta of 0 on its 32 images); run time within ±2 %.
- **Use:** with Addendum 12's rule (native vs module), the train runs on whichever of e5-native and V1f is not slower
  by more than 2 % on any of the four; ties go to V1f (option B, the lead's and collaborator's choice).

## Addendum 14 (9 Nov 18:5x, before either session runs): V1f first; the e5-native session only as a fallback

The rtl lane's simulation reads V1f identical to e5-native on all 32 gate images: values, end cycles and committed
instructions alike (delta 0). Its trace sweep equals V1's on all 92, which differs from its base only in option B's
own 8 raw-read tests. A vs B on silicon is therefore predicted equal, and a session to show that costs about 2.5 h of
board 2 the train needs. Order on board 2 after Capstone nab:
1. Addendum 13's V1f session (gate, then native gcc/imagick/nab/lbm, then the same through the ids55 module).
2. Addendum 12's e5-native session ONLY if V1f's gate does not pass (perf/train/qt-fallback.sh reads the V1f session's
   own log; dry-run on pass, fail and a stale pass line).
3. Board 2's chain as before.
- **Decision:** Addendum 12's native-vs-module rule, read on V1f. The bitstream for the Temporal train is V1f (option B),
  unless its quick tests fail T0, when the fallback session's e5-native readings decide.
- Addendum 13's A-vs-B comparison stays open. A V1f-vs-e5-native pair can be run after the deadline.

## Addendum 15 (9 Nov 19:3x, before any of it runs): a chain defect for non-vp2c bitstreams fixed; lifetimes-only module stages

**Defect, found by a deeper dry run.** perf/b3/board/spec-board.sh checks the kit with
`BOARD_BITSTREAM=caplifive_vp2c_<SB_BITSTREAM>`. Every bitstream sealed since stage B is named caplifive_<name> (E5c,
e5-native, the ids55 seals). So every module-route stage on them would have stopped with "is not the perf kit"
before the board: the quick tests on V1f and e5-native, and board 3's E5c perlbench fallback (Addendum 6). The earlier dry
runs stubbed the chain (GT_CHAIN=echo) and could not see it. Fix: a copy, perf/b3/board-x, whose kit check also accepts
`caplifive_<SB_BITSTREAM>`. The original stays as it is, because running chains read it. Controls: the four V1f stages
and an e5-native stage pass; a wrong name (e5nids55_b91439bf7 against V1f's kit) still STOPs. A deep dry run (driver →
chain → runner, only the board calls stubbed) reaches SB END rc=0 on all four.
**Module stages cut to the lifetimes arm** (spec/stages-qt, perf/train/cut-lifetimes.py: the spatial arm block removed,
every substitution counted, exes re-hashed). The native qt stages are lifetimes-only, and the two-arm module stages
would have cost about an hour more. The lifetimes block of each run.sh is byte-identical to its source. Exes: imagick
f50dc465183e4f1f + imagevalidate 8ecf59b59b9bc166 (Addendum 12 named the imagevalidate hash as imagick's; corrected
here), nab 092ff45c0c519283, lbm f0269eea1b555bd9. gcc's stage was already lifetimes-only.
Board 2's waiting sessions are relaunched with GT_CHAIN on board-x; no step had started.

## Addendum 16 (10 Nov 01:5x, before the re-run): V1f gate passed; native quick tests stopped at the bake, re-run after the fix

**What happened.** V1f's gate on board 2 passed (10 Nov 00:24, 32 PASS of 32, board left on
caplifive_e5nids55f_0ef470559). The native quick tests (qt-native-v1f.cmd) then stopped before either boot, on both
stages: "the bake saved no rootfs.cpio". The module quick tests went on as planned on the same bitstream.
**Defect.** The native kits' bake-manifest.sh saved Image and dtb only. Their board-run.sh relinks the private native
kernel on the bake's rootfs.cpio, so it stopped at that check. The kits' dry runs used BOARD_STUB=2, which takes a
prepared image that has a cpio, so the bake's real output was never checked. This applies to all ten native kits
(bkit36-native64{,-min}, bkit38-native-{mcf,perl}, bkit40-native-ids55-*, bkit41-native-v1f-{mcf,perl}). So board 2's
and board 3's native train steps (Addendum 5, nat-board.sh) would have stopped the same way and fallen back to C.5a.
**Fix.** bake-manifest.sh now also saves the baked rootfs.cpio, taken before the restore. The kits' SHA256SUMS are updated
(bake-manifest.sh 2504d7c03575f01c → 0de17ae89c3da809). Nothing else changed.
**Checked on a real bake, not a stub.** For the smoke stage, qt-gcc and qt-inl, each check ran without the board:
- the real bake;
- every manifest file present by hash in the saved cpio (smoke 17 of 17), and the replaced /capstone.ko;
- the real private-kernel relink on that cpio (BOARD_STUB=2 pointed at the real bake);
- the real firmware build: dom_stack gate PASS, monitor 4f51541 clean, size under the 26,000,000-byte cap (smoke
  19,565,576 B, qt-gcc 23,759,880 B, qt-inl 21,662,728 B);
- the pinned driver and V1f's sha256.
The first step that has not run is the driver itself.
**Re-run.** perf/train/qt-native-rerun.sh waits for the V1f session's fallback waiter, requires V1f's gate pass line and
a free board, then runs qt-native-v1f.cmd unchanged (sha 0c06eac3d67d16a2). Its dry run reaches DONE. Board 2's Plain
train (train-board4.sh) and native mcf train (nat-board.sh) are relaunched behind it with the same arguments. Neither had
started a step. This delays board 2's Plain by about 1.5 h.
**Decision rule unchanged** (Addendum 12 on V1f). If native fails again, module wins by default for the Temporal train
and the native failure is recorded as such.

## Addendum 17 (10 Nov 02:0x): the first native re-run stopped at the readback; the native quick tests move behind board 2's Plain

**What happened.** The module quick tests on V1f all passed (rc 0, specdiff 1/1 PASS on each):

| Test | Wall time (s) |
|---|---|
| gcc | 130.7 |
| imagick | 60.7 |
| nab | 991.1 |
| lbm | 1813.1 |

The module session ended at 01:56 by flashing board 2 back to bitstream 3, as every gt-run driver does. Addendum 16's
re-run then stopped at nt-run's readback ("board 2 is not on caplifive_e5nids55f_0ef470559"; nt-run does not flash).
No boot was spent. Its dry run stubbed that readback, so the bitstream the previous session leaves on the board could
not be seen.
**Board 2's Plain started meanwhile.** train-board4.sh (Plain imagick train, then xz c0 and c1) took the free board at
01:56. It is not interrupted.
**New order on board 2:**
1. Plain (running).
2. perf/train/qt-native-rerun2.sh. It flashes and gates V1f itself (v1f-gate.sh 2, the 32-image gate, output
   run-b2-qt2), then runs qt-native-v1f.cmd unchanged against that gate's log. This is the gate-then-native order that
   read V1f back correctly at 00:24. Dry run: on gate pass it reaches DONE; on gate fail it skips native.
3. nat-board.sh (native mcf train, relaunched with the same arguments, no step started).

The native-vs-module reading (Addendum 12's rule on V1f) moves to about 12-18 h from now. Meanwhile the Temporal train
goes on with the kits already queued; no queued stage depends on this reading.

## Addendum 18 (10 Nov 02:1x, the user's decision): board 2's Plain imagick stopped 11 minutes into its run; the native quick tests go first

The user decided this: a run that started recently may be stopped, and the next priority is which bitstream is optimal.
- **Stopped.** Board 2's Plain 538.imagick_r train, which began at 01:59:03 (step budget 90,000 s). It is stopped as one
  process group, the chain, driver and runner together. Its output directory is set aside as
  out-train4-b2-plain.aborted-1110, and no reading from it is used.
- **New order on board 2:**
  1. qt-native-rerun2.sh (Addendum 17): V1f flash and 32-image gate, then the native quick tests.
  2. Plain (train-board4.sh, the same two lists, from the start).
  3. nat-board.sh (native mcf train).
- **Cost.** Board 2's Plain ends about 1.5-2 h later, around Tue 23:30-24:00, against the Wed 20:00 deadline.
- **Next, if native wins on V1f.** The A-vs-B question for native (e5-native against V1f, Addendum 13) is the next
  candidate for a gap. It is not queued here.

## Addendum 19 (10 Nov 04:1x): native on V1f, first readings; nab was cut by the runner's silence limit; the kits fixed; imagick + nab + lbm again

**Readings (board 2, V1f, gate PASS 32/32 at 03:12, run-b2-qt2):**

| Test | Native wall (s) | Module wall (s) | Module in-VM (s) |
|---|---|---|---|
| gcc | 20.2 | 130.7 | 25.4 |
| imagick run | 20.6 | 60.7 | 23.0 |
| imagick validation | 11.2 | 21.1 | — |

- gcc's .s output is byte-identical between native and module. imagick's validation ended rc 0 on both.
- Module wall minus in-VM time grows with the executable's size, at about 7 s per MB: gcc 14.8 MB → 105 s, imagick
  5.2 MB → 38 s, nab 0.72 MB → 6 s, lbm 0.26 MB → 3 s. Its cause is not established. A whole-image hash at exit is
  the leading guess, because the unify kit's launcher prints `exit image=... sha256=...`. For nab and lbm it is under 1 %.
**Defect.** The runner stopped nab after 200 s of console silence (BAKED_IDLE_S=200 in the native kits' board-run.sh)
and recorded the vbatch rung as "wedged". The core was live: the debug apertures changed between reads. A native SPEC
stage prints nothing while a benchmark runs, so every native train (mcf and perl, queued on boards 2 and 3) would have
been cut the same way and fallen back to C.5a.
**Fix.** In the nine native kits in use, a stage with spec.sh gets BAKED_IDLE_S = the stage budget; the smoke keeps 200 s.
board-run.sh 311e827b230ec428 → adddd44ac0ddc01b, SHA256SUMS updated. bkit36-native64-min carries its own board-run.sh
and is not used; it is unchanged.
**Re-run (the user's rule: a run that started recently may be stopped).** Board 2's Plain imagick, started 04:04, is
stopped again; its directory is set aside as out-train4-b2-plain.aborted-1110b and its hub booking is released by id.
perf/train/qt-native-rerun3.sh then runs:
1. A V1f reflash only: sealed sha256 checked locally, flash, name read back. The gate is not repeated; it passed twice
   today on this board.
2. nt-run on qt-inl, the same imagick + nab + lbm stage, with the 03:12 gate log.
3. Plain and nat-board relaunched behind it, with the same arguments.

The dry run reaches DONE.
**Reading.** Addendum 12's rule on V1f: wall time per benchmark, native / module, over imagick, nab and lbm. The in-VM
column is reported beside it.

## Addendum 20 (10 Nov 04:4x): the re-run was cut again, this time by board-watchdog's entry-stall abort; fixed; run 4

Addendum 19's re-run passed its smoke boot. In the SPEC boot, board-watchdog.sh (ABORT_ON_ENTRY_STALL=1, 420 s)
aborted the runner 991 s in: "last share marker=SHA5:00000000, no SHA6 for 420s". The smoke steps that open the SPEC
boot leave SHA5 as the last share marker, the native route never prints SHA6, and nab runs silent. So the abort was
certain on any silent SPEC stage. No board transcript was written (DRIVER rc=130).
**Fix** in the nine native kits: a stage with spec.sh runs the watchdog with ABORT_ON_ENTRY_STALL=0. Its lines are kept
and only the abort is dropped. The NO-BOOT abort, which acts before the boot, is unchanged. The smoke keeps both limits.
board-run.sh adddd44ac0ddc01b → 24a7a9b634fef6ec. The watchdog's other kill paths act only before the boot (NO-BOOT).
**Run 4.** qt-native-rerun4.sh is rerun3 plus three changes:
- It requires both fixes in the kit.
- It skips the flash when V1f reads back resident, which it does now.
- It stops board 2's waiting Plain and nat-board scripts before either started a step, and relaunches them behind it.
The dry run reaches DONE.
Each of these defects stopped a native session on board 2 tonight before the native route itself was exercised:
- the bake saved no cpio;
- the runner did not flash;
- the runner's silence limit;
- the watchdog's entry-stall abort.
None of them is a native-route result.

## Addendum 21 (10 Nov 05:5x): native vs module on V1f, read; native wins on all three

Run 4 (perf/native-qt/run-b2-v1f-inl-1110-044054, kit bkit41-native-v1f-mcf, V1f resident, gate run-b2-qt2 PASS):
smoke 9/9 PASS; SPEC boot 4 commands, all rc 0, 3 @@SPEC-DONE. Module: out-qt-b2-v1f-mod (kit bkit41-ids55-unify),
same board, same bitstream, same lifetimes executables.

| Test | Native wall (s) | Module wall (s) | native / module | Module in-VM (s) | native / module in-VM |
|---|---|---|---|---|---|
| imagick | 20.59 | 60.74 | 0.339 | 23.02 | 0.894 |
| nab | 894.17 | 991.08 | 0.902 | 984.67 | 0.908 |
| lbm | 1658.05 | 1813.13 | 0.914 | 1809.75 | 0.916 |
| gcc (first number) | 20.17 | 130.65 | 0.154 | 25.38 | 0.795 |

**Outputs.** Native and module outputs are identical by hash:
- imagick test_output.tga and test_validate.out; its validation is rc 0 on both;
- nab's and lbm's output files;
- gcc's .s.

The only differences are the .err files, which on the module carry its statistics lines, and imagick's reference .tga.
The reference file is hashed only in the native run directory and equals the stage's ref copy (d7e4e331).
**Decision (Addendum 12's rule).** Native is faster on 3 of 3 and slower on none, by wall time and by the module's
in-VM time alike. Native on V1f takes the remaining Capstone train. Option B costs nothing on the module route
(Addendum 17) and is the bitstream these readings come from.
**Not established here:**
- Native over a long run. These are test-size runs of 20-900 s; a train runs for hours.
- The ratios to Plain. The Plain test readings come from other sessions and kits, not same-board pairs.

## Addendum 22 (10 Nov 08:5x, before any of it runs): the remaining Capstone train on V1f + native; no running step is stopped

The user's decision: as many benchmarks as possible on the optimal configuration (V1f + native, Addendum 21), every
problematic one among them, and the deadline holds. Running steps are not interrupted. Plain mcf (board 1), Plain imagick
(board 2) and Plain lbm (board 3) run on. Only waiting wrappers are replaced, and none of them had started a step:
train-board3 3882166 (board 1), nat-board 1097298 (board 2), nat-board2 576476 and train-board4 576649 (board 3).
**Already on the record, module route on C.5a:** nab (79,730 s, specdiff 2/2), x264 (5/5, 3/3) and gcc train01
(Addendum 3). These have no V1f run here. A native nab train would take about 20 h, and no slot holds that.
**Native stages** (native01/board-stages, SHA256SUMS ok):
- perl-train and mcf-train: Addendum 5's, QEMU-twinned by the native lane.
- nt-imagick-train and nt-lbm-train: their exes equal by hash the ones that ran and passed on V1f in Addendum 21
  (imagick_r f008168e93193db3, imagevalidate 6e5613cc944bb95d, lbm_r ed43d5cc37d18396).
- **Kits:** perlbench runs on bkit41-native-v1f-perl (524,288 records). mcf, imagick and lbm run on bkit41-native-v1f-mcf
  (262,144 records; imagick's test run used about 1.2 k nodes). Both kits load the same module (f2a2645cc570dcc6).
**Driver: perf/train/nat-v1f.sh.**
- Each native step:
  1. waits for a free board;
  2. reads the resident bitstream back, and flashes V1f only if another bitstream is resident (sealed sha256, the flash
     tool's checks, readback);
  3. runs nt-run-v1f.sh: a smoke boot, then the train boot.
- **Pass condition:** nt-run rc 0, every @@SPEC-END rc=0, as many markers as the plan's commands plus validations, and
  @@SPEC-DONE.
- **Deadline guard:** before every step and every fallback, a step whose nominal hours do not fit before Wed 20:00 is
  SKIPPED, not started. Native timeouts are capped at the deadline.
- **Defect fixed along the way:** nt-run's default train budget is 20,000 s. nat-board.sh never set it, so the queued
  native mcf and perlbench would have been cut at 5.6 h. Each step now sets its own: perlbench 90,000 s, mcf 76,000,
  imagick 68,000, lbm 37,000.
- **Dry runs** (scratchpad, every board call stubbed, all ending DONE):
  - native pass;
  - native fail into the fallback: mcf4k; e5c, with the E5c gate failing into C.5a unify;
  - a late start that skips perlbench;
  - a late fail with the timeout capped.
**No 32-image V1f gate on boards 1 and 3.** V1f passed it three times on board 2 from the same sealed file. On each
board, the flash checks, the readback and the native smoke boot (Tier 1) still run first. This saves about 2 board-hours.
**Plain xz c1 moves from board 2 to board 3.** Board 2 skips it (SKIPPED in out-train5-b2-plain-c1), and board 3 runs
the same stage first (list perf/train/b3pc1-plain, out-train5-b3-plain-c1). Plain still runs first on each board.

| Board | After the running Plain (est. end) | Then | Nominal end | Capstone at +35 % |
|---|---|---|---|---|
| 1 | Plain mcf (~Tue 19:00) | native imagick 13.1 h; Cap xz c0 (C.5a nset) 6 h | Wed ~14:00 | imagick Wed ~12:40; xz c0 then skipped |
| 2 | Plain imagick (~18:00); Plain xz c0 (~23:00) | native mcf 14.7 h (fallback C.5a 4K q32 15.7 h) | Wed ~13:40 | Wed ~18:50 |
| 3 | Plain lbm (~10:00) | Plain xz c1 2.5 h; native perlbench 17.5 h (fallback E5c 19.6 h); native lbm 7.5 h; Cap xz c1 (C.5a nsetfw) 2.9 h | Wed ~16:30 | perlbench Wed ~12:10; lbm fits; xz c1 skipped |

- **Predictions:** N0: each native step passes as above, and its outputs equal the reference (checked afterwards, as in Addendum 21). N1: native time = 0.85-0.95 × the module estimate
  in the schedule.
- **A native failure:** falls back if the guard allows. Otherwise that benchmark reports its test-size number, marked
  as such (rule T0). xz has no native stage and stays on the module route on C.5a.

## Addendum 23 (10 Nov 08:4x, before it runs): native gcc train01 on V1f, board 2 after native mcf; the four native stages checked on a real bake

**Real bake, no board** (bake, private-kernel relink on that bake, firmware with its dom_stack gate, the 26,000,000-byte cap):

| Stage | Kit | cpio (B) | Firmware (B) | Gate |
|---|---|---|---|---|
| perl-train | bkit41-native-v1f-perl | 24,154,112 | 23,759,880 | PASS |
| mcf-train | bkit41-native-v1f-mcf | 13,687,808 | 19,565,576 | PASS |
| nt-imagick-train | bkit41-native-v1f-mcf | 19,275,264 | 19,565,576 | PASS |
| nt-lbm-train | bkit41-native-v1f-mcf | 12,782,592 | 19,565,576 | PASS |
| nt-gcc-train01 | bkit41-native-v1f-mcf | 27,479,040 | 23,759,880 | PASS |

**gcc train01 native.** gcc's train number is on the module route (C.5a, 1.196x Plain, Addendum 3), at the 1.20 bar.
- **Stage:** native01/board-stages/nt-gcc-train01.
- **How it was made:** make-native-stage.py (llvm-capstone 727a28aec168), run on stages-train-fair/502.gcc_r-train01-capstone-lifetimes-devmk-dcnofp. Then the module-only lines were removed, each counted and asserted:
  - step_budget_ns, resume_quantum, retire_exit_sweep and arena_pin, and their restore;
  - the gtsnap calls.
- **Result:** its run.sh equals the native test stage's (qt-gcc) except for the input names.
- **Exe:** qt-gcc's cpugcc_r_base.native, df93c433bb1a5418. It ran and passed on V1f in Addendum 19.
- **Not checked:** no QEMU twin. The board run is the first run of this input on the native route, and its outputs are checked against the reference afterwards.
- **Where it runs:** board 2, after native mcf (a second nat-v1f.sh waits for the first). Timeout 10,000 s; 1.5 h needed by the deadline guard. Nothing else moves.
- **If it fails:** gcc keeps its module train number.

## Addendum 24 (10 Nov 12:5x): Addendum 22's native timeouts were not in force; an external watcher enforces them

**Defect.** nat-v1f.sh passes each native step's timeout to nt-run-v1f.sh as NT_TRAIN_TO, and the kit's board-run.sh
multiplies it by BOARD_TIMEOUT_MULT=4 (board.env). The running perlbench train boot reads `idle limit 360000 s` in its
SEQ and its runner is `timeout 361800`. So Addendum 22's "native timeouts are capped at the deadline" was false: a wedged
native step would have held its board past the deadline and lost the steps after it. Found by reading the running boot's
SEQ. It has cost nothing so far: perlbench command 0 started at 12:07 and is running.
**Fix, with no running script edited.** perf/train/cap-enforce.sh enforces the pre-registered per-step caps (perlbench
90,000 s, mcf 76,000, imagick 68,000, lbm 37,000, gcc 10,000, each already capped at the deadline by nat-v1f.sh).
- **Trigger:** a native step's train boot (nt log "BOOT 2 train BEGIN") has run longer than its cap.
- **Action:** the watcher kills the process group of the `timeout ... run_baked_rungs_fpga` runner whose ancestor
  board-run.sh names that step's output directory. Nothing else is touched.
- **Effect:** nt-run logs BOOT 2 END and unbooks. nat-v1f reads the step as FAIL and goes on through its deadline guard.
One watcher runs per board and ends with that board's nat-v1f processes.
**Controls:**
- Negative: on board 3's live state it takes no action.
- Dry, with the clock set to Wed 15:00: it names exactly the live perlbench runner's pgid (1476878) and does not kill it.
- Positive: on a fake step with a fake runner tree it kills the runner (rc 143), and the parent script continues.

## Addendum 25 (10 Nov 17:1x, the user's decision): xz command 0 goes to the end of every queue

xz command 1 already has its Plain number (8,736.0 s, specdiff 1/1), and its Capstone run (board 3) is queued, so xz
can be reported on command 1 if command 0 does not complete. Command 0 now runs last on both arms, after every native
step.
- **Plain xz c0** leaves board 2's running Plain chain before it starts: an output directory
  out-train4-b2-plain/557.xz_r-train-plain-rv64-link-c0r with a SKIPPED marker makes chain-board.sh skip it
  (chain-board.sh:26). Plain imagick, still running, is not touched.
- **It is requeued** as the last step on board 2, after native mcf and native gcc. A third nat-v1f.sh waits for the gcc
  waiter; it uses the same stage, kit (bkit27-gtq) and driver (gt-run-ga2.sh). Its list is perf/train/b2pxz0-plain, its
  output out-train6-b2-plain-xz0, and the deadline guard needs 5 h.
- **Capstone xz c0** (board 1, after native imagick) is already its board's last step and is unchanged.
- **Dry run:** the new step starts when given Wed 10:30 and is SKIPPED when given Wed 16:00.
- **Board 2:**
  - Native mcf now starts after Plain imagick, about 18:00, not about 23:00.
  - Nominal ends: native mcf Wed ~08:40, gcc ~10:10, Plain xz c0 ~15:00.

## Addendum 26 (11 Nov 06:1x, before its board call): every native/Plain ratio also crosses a bitstream; Plain on V1f at test size

**The confound.** Every Plain train run is on C.5a (caplifive_vp2c_1fc2cd5aa; out-train*-plain SEQ files). Every native
run is on V1f (caplifive_e5nids55f_0ef470559). So each native/Plain ratio is also a ratio between two bitstreams. The
same is true of the test-size native ratios (Plain at test ran on C.5a, results/2026-11-07-plainreuse Addendum 4). The
module-route train ratios are not affected, because both arms ran on C.5a. That includes x264.
- No Plain run exists on any V1f or E5 bitstream. All `spec/board/out-*` SEQ files were searched.
- The RTL between the two differs in more than the capability paths: 25 files under core/, including wt_dcache_ctrl,
  load_unit, the frontend and cva6_mmu. 213378a11 changes when a load through a 2 MiB page takes the FPGA's
  non-idempotent stall.
- Precedent: Plain moved within ±2 % between the 3 Nov bitstream and C.5a (plainreuse R5). That is evidence about those
  two bitstreams only.

**Native results so far** (V1f, board as listed, against the C.5a Plain train run):
- perlbench, board 3: 5/5 rc 0, specdiff 6/6. 54,021.8 s against Plain 48,206.5 s (board 1): 1.121.
- imagick, board 1: 2/2 rc 0, specdiff 1/1. 38,251.9 s against Plain 43,517.9 s (board 2): 0.879. Every output hash
  equals Plain's, train_output.tga included. The twin read prints DIFFER only because the native stage carries one
  extra file, 8811215d-train_expected.tga, whose hash equals train_output.tga's.
- imagick at 0.879 is not reported as a gain. It reverses the test-size reading (1.058). The calibration below tells
  whether the bitstream accounts for it. If it does not, the Plain imagick train run (hart CPI 3.66) is the next thing to
  check.

**Calibration.** Board 1, after its last train step (Capstone xz c0, ~14:00). Driver perf/train/cal-v1f.sh:
1. Wait for board 1's nat-v1f.
2. Make V1f resident: readback, flash if needed, readback.
3. Run perf/b3/board-x/chain-board.sh over perf/train/cal-v1f.list. The list holds the three Plain-with-cache test
   stages of plainreuse Addendum 4, unchanged: imagick, nab and lbm.

The kit is bkit41-ids55-unify. Its board.env equals bkit27-gtq's except BOARD_BITSTREAM, with the same monitor
(d9da397), firmware recipe and wrap. Its module and launcher are the ids55 ones that V1f's known-good control needs.
Plain does not use them.

The pair is same-board and same-stage against Addendum 4's C.5a readings on board 1: imagick 19.46 s, nab 862.99 s,
lbm 1,895.0 s.

| # | Prediction |
|---|---|
| C1 | V1f/C.5a for Plain within ±1 % for nab and lbm, and ±3 % for imagick (20 s, startup-dominated) |

Basis for C1, from readings already made: module-route code moved by at most 0.1 % between the two bitstreams at test
size. For lbm, native/module on V1f was 0.914 (board 2) and native V1f/Plain was 0.875, so module V1f/Plain = 0.957,
against module C.5a/Plain = 0.956. For nab the same chain gives 1.149 against 1.148. This is evidence about Capstone
code; the calibration measures Plain code.

**Reading rule.** If C1 holds, the native train ratios stand against C.5a Plain. If C1 fails, each native ratio is
reported twice: once against C.5a Plain, and once corrected by the measured factor for that benchmark, or by the
geomean factor where none was measured. The correction is labelled as a correction, never as a same-bitstream
measurement.

**Dry run** (perf/train/cal-dry; every board call stubbed): it reaches the chain, and spec-board's kit and stage checks
pass for real on all three stages. A pre-registration that is not on origin STOPs. An existing output directory STOPs.

## Addendum 27 (11 Nov 09:3x, before it runs): native gcc train01 was stopped by mistake; it runs again after Plain xz c0

**What happened.** The user said gcc already has a train number and should not be measured again. Native gcc train01
on board 2 (Addendum 23, perf/native-train/v1f-gcc-b2) was stopped at 09:14. The stop used cap-enforce's method:
TERM to the runner's process group, then KILL after 30 s. At that point command 0 had run about 76 minutes and had not
ended, so the run has no reading. Minutes later the user withdrew the request: native gcc runs, because it is on the
new bitstream. Plain xz c0 had already started on board 2 at 09:14:20 (Addendum 25) and is not interrupted.

**Re-run.**
- It runs as board 2's last step, after Plain xz c0, with the same stage (native01/board-stages/nt-gcc-train01), kit
  (bkit41-native-v1f-mcf), timeout (10,000 s, capped at the deadline) and deadline-guard need (1.5 h).
- The driver is a new nat-v1f.sh. It waits for the xz waiter, and its label gcc2 writes perf/native-train/v1f-gcc2-b2.
  The stopped run's directory is kept as it is.
- A new cap-enforce watches it.
- V1f is made resident first: the step reads the bitstream back and flashes V1f if the xz driver left another one.

**Prediction:** Addendum 23's (rc 0, specdiff PASS).

**Reporting rule, unchanged from Addendum 23:**
- If it passes, gcc reports the native number, whether it is above or below the module's 1.196, with the module
  number beside it.
- If it fails or is skipped by the deadline guard, gcc keeps the module number.
- The stopped run is not a reading.

## Addendum 28 (11 Nov 10:4x): RETRACTION. The native/Plain train ratios are not fair, because the Plain arm pays a 9-11 % harness heartbeat that the native kit does not run

**Retracted:**
- Every native-route train ratio as a fair Capstone/Plain number: perlbench 1.121, imagick 0.879 (Addendum 26), and
  mcf 1.062 and lbm 0.840 (recorded since).
- The run times, outputs and specdiff verdicts stand. Only the denominator is unfair.

**Mechanism.**
- The Plain kit's batch script (boardvm.py:200-206) runs a heartbeat every 30 s. Each beat is about six process
  spawns (`wc`, `tail`, `base64`), and the arms share it.
- Measured cost: 3.1-3.7 s of hart time per ~35 s beat, i.e. 8.9-10.8 % of the hart, from three sources:
  - mcf out-mx-b2: GT-SNAP window minus rusage, 819 s of 9,087 s;
  - gcc train: four runs, 9.3-10.8 %;
  - lbm test: `times`, 173 s of 1,899 s.
- Direct check, imagick r10 Plain: repetitions containing a beat take 19.05-20.06 s, against 16.36-16.42 s for those
  without one.
- The native kit (bkit41-native-v1f-*, nt-run spec.sh) has no heartbeat: 0 `@@HB` lines in the native lbm and
  imagick train transcripts, against 712 in Plain lbm train.
- The native and stock kernels have the same tick configuration (HZ=100, HZ_PERIODIC, PREEMPT_NONE, no THP).
- The module route is not affected, because both arms run the same gate.sh. Capstone pays the beats in ucyc, outside
  its in-context counters. Its ratios stand: x264 0.882, nab 1.088 and gcc 1.196.

**Corrected estimates.** Each is labelled a correction, not a measurement: native / (Plain × (1 − f)), with f in
[0.089, 0.108].

| Benchmark | Correction |
|---|---|
| perlbench | 1.230-1.256, above the 1.20 bar |
| mcf | 1.166-1.191 |
| imagick | 0.965-0.985 |
| lbm | 0.922-0.942 |

**Ruled out for the sub-1.0 readings, with evidence:**
- Misaligned-access emulation. A QEMU plugin counted 0 misaligned accesses in Plain lbm, imagick, mcf, nab and x264
  at test size. Its positive control counted 1,000,003 of 1,000,003. On this firmware a misaligned user access is a
  SIGBUS, not an emulation.
- Bitstream. Every Plain and module run was read back as C.5a. Capstone lbm's in-context cycles are 34.57 G on C.5a
  and 34.55 G on V1f.
- Huge pages. The module refuses huge mappings, and THP is off in both kernels.
- Executables. The hot loops do the same FP and memory work, plus 0-4 integer instructions for Capstone. lbm train is
  86.0 G instructions for Plain against 88.5 G for Capstone.

**Still open:**
- x264's module ratio, 0.882 on train and 0.897 on test, with the same harness in both arms. Plain's user time
  exceeds Capstone's in-context time by about 8-11 %. The candidate is the per-tick cost: Plain takes 100 live ticks
  per second, while the context masks them and replays them in batches. That cost is unmeasured.
- lbm, where Plain user time is 1,589.5 s against 1,383 s in context, likewise.

## Addendum 29 (11 Nov 10:5x, before it runs): the heartbeat's cost measured per benchmark, Plain with and without it, on board 1 after the calibration

**Question.** How much of each Plain reading is the batch heartbeat? This is the f that Addendum 28 bounded at
0.089-0.108 from other runs.

**Arms.** One variable: the heartbeat. Plus the runner idle limit, which a silent step needs.
- **No heartbeat (chain A):**
  - run-stage.py from a copy of the board tools (perf/nohb/tree). Its boardvm.py heartbeat subshell is replaced by
    `( : ) & hb=$!`; nothing else changes.
  - Kit bkit41-ids55-unify-nohb: bkit41-ids55-unify with board-run.sh `BAKED_IDLE_S=200` changed to `$TO`, and
    SHA256SUMS regenerated.
- **With the heartbeat:**
  - perl c0 runs in chain B: the standard tools and bkit41-ids55-unify.
  - imagick, nab and lbm use Addendum 26's calibration readings, which are the same board (1), bitstream (V1f), kit
    and stages.
- **Stages:**
  - The plainreuse imagick, nab and lbm test stages, unchanged.
  - perlbench test command 0 only (makerand.pl, about 500 s), at perf/nohb/stages/500.perlbench_r-test-plain-rv64r-c0.
    It is the plainreuse stage with the command-1 block, its two output files and its host validation removed. The
    workload is unchanged.
- **Order:** chain A runs perl c0, imagick, nab and lbm; then chain B runs perl c0. About 2 board-hours.

**Metric.**
- The per-beat cost c = (T_hb − T_nohb) / beats, where beats is the number of @@HB lines inside that command's window
  in the heartbeat transcript.
- The steady-state share is f = c / beat period. The period is about 35 s; each run's own value is measured.

**Predictions:**

| # | Prediction |
|---|---|
| N1 | 0 @@HB lines in every chain-A transcript; outputs and specdiff as in the heartbeat runs |
| N2 | c in [3.0, 3.8] s for each benchmark. Predicted times: perl c0 ≈ 452 s, imagick 16.2-16.6 s, nab ≈ 780 s, lbm ≈ 1,720 s |

**Reading rule.**
- Each measured f replaces Addendum 28's range for that benchmark's corrected native ratio: perl's f from perl c0,
  lbm's from lbm, imagick's from imagick.
- mcf keeps the range.
- Everything stays labelled a correction. A same-environment native/Plain pair at train size still needs
  board time after the deadline.

**Dry runs** (scratchpad nohb-dry; board calls stubbed):
- Both chains reach every stage, and spec-board's kit and stage checks pass.
- A fake pre-registration stops the run; an existing output directory stops it.
- Generating the real batch with the copy, against a recorded transcript, gives a gate.sh with 0 heartbeat loops. The
  original tools give 1.

## Addendum 30 (11 Nov 12:0x, before it runs): Addendum 29's chain A lost three of four stages to the watchdog; re-run with its entry-stall abort off

**What happened.** Chain A (kit bkit41-ids55-unify-nohb) read imagick: 16.13 s, specdiff 1/1 PASS, 0 @@HB lines. perl c0,
nab and lbm ended rc 2 with no SPEC step in the transcript. In each, board-watchdog.sh printed `ENTRY-STALL ... last share
marker=SHA5, no SHA6 for 420s` at 735-795 s and killed the runner. The kit starts the watchdog with ABORT_ON_ENTRY_STALL=1
and ENTRY_STALL_S=420. In the heartbeat kit the 30 s beats keep the console from ever being quiet for 420 s, so that abort
never fires on a Plain batch. Without the heartbeat, any command quiet for 420 s reads as an entry stall. Addendum 29's
dry runs stubbed the runner and the watchdog, so they could not show this.

**Change.** Kit bkit41-ids55-unify-nohb2: the nohb kit with `ABORT_ON_ENTRY_STALL=0` on the watchdog line, plus a board.env
comment and a regenerated SHA256SUMS. The watchdog still runs and still logs QUIET/STALE. The runner's own idle limit
stays at the step budget, as in Addendum 29. Driver perf/nohb/nohb2-v1f.sh, which is nohb-v1f.sh with:
- chain A only, over nohb2.list (perl c0, nab, lbm);
- output spec/board/out-nohb2-v1f-b1;
- the kit diff check allowing the abort line too;
- a check that the kit really has the abort off.

It waits for the Addendum 29 driver (chain B, perl c0 with the heartbeat) to exit, then runs on board 1, V1f.

**Predictions and reading rule.** Addendum 29's, unchanged: N1, N2 (perl c0 ≈ 452 s, nab ≈ 780 s, lbm ≈ 1,720 s). The
imagick reading is Addendum 29's chain-A one. The three killed stages give no reading.

**Dry runs** (scratchpad nohb2-dry; board calls stubbed):
- All three stages are reached, with kit nohb2.
- A fake pre-registration stops the run, and so does an existing output directory.
- Pointing the driver at the old nohb kit stops it: "keeps the entry-stall abort".

## Addendum 31 (11 Nov 12:4x, before it runs): does RVC explain Plain's higher CPI? Plain test with and without compressed instructions, board 3

**Question.** Plain user code runs at an 8-15 % higher CPI than the same work inside Capstone contexts (lbm, x264,
imagick; Addendum 28's open item). The same-work QEMU instruction ratio, Capstone in-context over Plain user, is 1.026 for
lbm train, 1.039 for imagick and 0.994 for x264. One difference has not been measured: Plain is `-march=rv64gc`, and 41-48 % of
its static instructions are 16-bit (in lbm's LBM_performStreamCollideTRT, 207 of 833). Capstone code has no compressed
encodings. On CVA6, mixed 16/32-bit code can cost fetch bubbles: 32-bit instructions that straddle fetch blocks, and
2-byte-aligned branch targets. The other candidate, live 100 Hz ticks, is unchanged by this pair.

**Arms.** One variable: the program's code is compiled with `-march=rv64gc` (the record) or `-march=rv64g`.
- RVC: the record's Plain test stages, perf/plainreuse/stages/{538.imagick,544.nab,519.lbm}_r-test-plain-rv64r. Their
  executables are the plain-rv64r builds (cache musl). Their plan.json still lists the earlier plain-rv64 hashes; the
  files were swapped on 8 Nov without updating it. Noted here, and not changed.
- No RVC: copies with only the executables swapped for plain-rv64rg. The build is build-r3's, with `-march=rv64g` in
  perf/norvc/plain-cc-reuse-g, label plain-rv64rg, and the same clang, flags, cache musl and libgcc. musl's libc.a stays
  rv64gc, so the remaining 2-byte instructions are libc code (printf_core, __floatscan, vfscanf...): imagick 3.0 % of
  static instructions, nab 13.5 %, lbm 35.6 %. lbm's hot function has 0.
  - Executables: lbm 1dc62d7766773cb2, imagick 662609e82596f715, imagevalidate 2df726fcc87fac38, nab d3f2b014b73035ae.
  - plan.json differs only in `executables`.
- Board 3, C.5a, kit bkit27-gtq: the Plain train kit, with the heartbeat in both arms. Driver perf/b3/gt-run-ga2.sh
  (flash C.5a, chain, flash bitstream 3 back), list perf/norvc/b3/b3.list. The arms alternate per benchmark: imagick,
  nab, lbm.

**Predictions:**

| # | Prediction |
|---|---|
| R1 | Both arms rc 0 and specdiff PASS; the no-RVC instruction count in QEMU is within 0.5 % of the RVC one |
| R2a | If RVC explains the residual: no-RVC lbm and nab run ≥ 5 % faster than RVC; imagick moves the same way |
| R2b | If it does not (ticks or something else): every pair within ±2 %. The RVC arms reproduce the C.5a record within ±2 % (imagick 19.46 s, nab 862.99 s, lbm 1,895.0 s) |

**Reading rule.**
- Under R2a, report the CPI share RVC accounts for, and flag every Plain-relative ratio as including a Plain-side ISA choice.
  Whether the fair denominator is rv64gc or rv64g is the lead's call, not this record's.
- Under R2b, RVC is ruled out for the residual.
- Nothing else is re-labelled.

**Dry runs** (scratchpad rvc-dry):
- The driver and chain-board reach all six stages, with only the board calls stubbed.
- board-env-check passes.

## Addendum 32 (11 Nov 13:1x, before it runs): every open fairness hypothesis in one place, and two more board checks (board 1)

**Status of each hypothesis.** The question in every row: does the hypothesis make a Capstone/Plain ratio wrong?

| # | Hypothesis | Status |
|---|---|---|
| F1 | Native kit runs no heartbeat, Plain kit does (9-11 %) | CONFIRMED; native ratios retracted (Add. 28), measured per benchmark (Add. 29/30) |
| F2 | V1f vs C.5a bitstream changes Plain | RULED OUT (Add. 26, C1: 1.006, 0.9994, 0.9967) |
| F3 | Misaligned accesses emulated on Plain only | RULED OUT (0 in every Plain test; control 1,000,003 counted) |
| F4 | Huge pages or THP for one arm | RULED OUT (capstone_vm refuses pmd_leaf; THP off) |
| F5 | Capstone does less work (compiler, libm, flags) | RULED OUT as a Capstone advantage: same-work QEMU instructions Capstone/Plain 1.026 (lbm train), 1.039 (imagick), 0.994 (x264) |
| F6 | Capstone's clock runs slow while ticks are masked | RULED OUT: host time minus guest uptime is 1,694 s (Capstone x264 train) vs 1,669 s (Plain), a 0.06 % difference over 40,000 s |
| F7 | Plain's RVC code costs CVA6 fetch cycles | TESTING (Add. 31, board 3). The RTL read finds one RVC-only bubble: a redirect to a 2-mod-4 target whose first instruction is 32-bit. It predicts a small effect |
| F8 | Plain pays live 100 Hz ticks through the M-mode monitor (no sstc; rdtime trapped) plus cache refill; Capstone masks ticks for its quanta | TESTING here (P1) |
| F9 | The Capstone arm pays the heartbeat differently from Plain, so the module-route ratios (x264, nab, gcc) are not symmetric after all | TESTING here (P2) |
| F10 | Allocator: Plain-with-cache (mallocng + mapping cache) has a different footprint or kernel time | OPEN, unmeasured. Plain xz c0 is OOM-killed at anon-rss 902 MB, while Capstone c0 fits in 871 MiB. Plain test sys share: lbm 7.2 %, imagick 9.9 % |

**P1, the tick cost on Plain.** perf/tick/hiccup.c (plain-cc, rv64gc) runs in place of imagick_r in a copy of the imagick test stage
(perf/tick/stage-hiccup; the executable is 9a81ac642924cd07, plan.json updated). Board 1, V1f, kit bkit41-ids55-unify
(the calibration kit, heartbeat on). Driver perf/tick/tick-v1f.sh, which is cal-v1f.sh with list perf/tick/tick.list.
- Phase A (60 s): a tight rdcycle loop. Every step over 400 cycles is time spent away from the task. Gaps under 1 ms
  are counted as ticks, gaps of 1 ms or more as heartbeat or other.
- Phase B (60 s): 256-load passes over a 16 KiB working set. The extra time in the four passes after a gap is the refill
  cost. It is a lower bound for SPEC, whose working sets and code are larger.
- Predictions:
  - F8 explains the residual (8-15 % CPI): small gaps at about 100/s, and stolen plus refill at least 5 % of cycles.
  - F8 is minor: stolen plus refill under 2 %.
  - Either way, the measured MHz is 25.0 ± 0.1.

**P2, the heartbeat on the Capstone arm.** Capstone lbm test, module route (stages-eval-b1/519.lbm_r-test-devmk, the
out-all2-b1 run: 1,809.88 s, 104 @@HB). It runs on kit bkit23-all2-b1 with the standard tools (B), and on kit
bkit23-all2-b1-nohb with the no-heartbeat tools copy perf/nohb/tree (A). The nohb kit differs from bkit23-all2-b1 by
the runner idle limit = step budget, the watchdog entry-stall abort off, and a board.env comment, with SHA256SUMS
regenerated. Driver perf/nohb/capnohb.sh (from nohb2-v1f.sh), board 1: flash C.5a if it is not resident, then A, then B.
- Predictions:
  - F9 false: B − A per beat is in Plain's 3.0-3.8 s band. The Capstone arm pays the heartbeat as Plain does, and the
    module-route ratios stand as symmetric.
  - F9 true: the per-beat cost is outside that band. Then the module ratios carry an asymmetric correction, which gets
    reported.
  - B reproduces 1,809.88 s within ±2 %.

**Order on board 1:** Addendum 30's re-run (running), P1, P2. Then board 1 stays on C.5a.

**Dry runs** (scratchpad tick-dry, capnohb-dry; board calls stubbed):
- Both drivers reach every stage, and a fake pre-registration stops each of them.
- capnohb stops on an existing output directory.
- P1's executable was run in QEMU before the board.

## Addendum 33 (11 Nov 13:3x): results of Addenda 29/30 (the heartbeat's share, measured), xz, and the cheap fairness checks

**The heartbeat's share on Plain test** (board 1, V1f; the heartbeat arm is Addendum 26's calibration, or chain B for perl c0):

| Benchmark | With HB | Without HB | Beats in the step | c per beat | f = (T_hb − T_nohb) / T_hb |
|---|---|---|---|---|---|
| perl c0 | 498.04 s | 450.95 s | 26 (period 19.2 s) | 1.81 s | 0.0946 |
| imagick | 19.58 s | 16.13 s | 1 | 3.45 s | 0.176 (one beat in 19 s, not a steady-state share) |
| nab | 862.44 s | 775.94 s | 25 (period 34.5 s) | 3.46 s | 0.1003 |
| lbm | 1,888.68 s | 1,713.29 s | 55 (period 34.3 s) | 3.19 s | 0.0929 |

- **Checks.** Every arm is rc 0 with specdiff PASS. N1 holds: 0 @@HB lines in every no-heartbeat transcript.
- **N2 holds for imagick, nab and lbm, and fails for perl c0.** perl's beats come every 19.2 s rather than about 35 s, and
  each costs 1.81 s, but its share, 0.095, is the same as nab's and lbm's. The predicted no-heartbeat times were met:
  perl 452 s (read 450.95), nab about 780 s (775.94), lbm about 1,720 s (1,713.29). imagick's 16.2-16.6 s read 16.13 s.
- Beats are the @@HB lines between @@STEP-BEGIN spec and the step's output. The step's output is printed after it
  ends, so the beats interleave before it.
- **Corrected native train ratios, raw / (1 − f).** These replace Addendum 28's ranges and are still labelled
  corrections:
  - perlbench 1.121 → **1.238** (perl f);
  - lbm 0.840 → **0.926** (lbm f);
  - mcf 1.062 → **1.171-1.180** (lbm-nab f range);
  - imagick 0.879 → **0.969-0.977** (lbm-nab f range).
- **One deviation from the reading rule.** imagick's own f is not used: its test step holds a single beat, at the start,
  so 0.176 overstates the steady share of a 12-hour run. The lbm-nab steady range is used instead.

**xz.**
- **c1:** Capstone 8,730.5 s (out-train5-b3-nset-c1, specdiff PASS) against Plain 8,735.95 s: **0.999**, module route, both
  arms with the heartbeat.
- **c0:** NO READING.
  - Plain was killed at 9,764 s (rc 137, "Killed", no @@STEP-BOUND).
  - A board-memory twin (pinwheel, 936M, cma=256M), run as a diagnostic copy of the stage that prints meminfo and dmesg
    around xz with the workload unchanged, reads "Out of memory: Killed process (xz_r_base.plain) total-vm:954124kB,
    anon-rss:902012kB" on MemTotal 921,484 kB.
  - Plain xz c0 does not fit a 1 GiB board, while Capstone c0 did (12,159.09 s, peak_pages 222,976). The difference is
    F10, open.

**Fairness checks run today (Addendum 32's table):**
- **F5, compiler.** The SPEC configs have the same OPTIMIZE (-O2) and PORTABILITY lines.
  - The cc1 sets on lbm.c share -O2, -ffp-contract=on, -vectorize-loops and -vectorize-slp.
  - Capstone alone has: -fno-jump-tables, -mframe-pointer=non-leaf, -target-abi lp64 (FP arguments in integer
    registers), static, +movc-keeps-integer-source, -capstone-gp-free and -capstone-image-gp.
  - Plain alone has: rv64gc, lp64d, PIE codegen, -mframe-pointer=none, -fno-signed-char and unwind tables.
  - Plain lbm.c compiled by Capstone's compiler (llvm-c1 724bf0bea11e) and by Plain's (cc-prmerged fca004e7e063)
    disassembles byte-identical (2,177 instructions). The compiler commit does not change generic optimisation here.
- **F6, clock.** RULED OUT. Host time minus guest uptime is 1,694 s for Capstone x264 train and 1,669 s for Plain.
- **F11, NEW.** The devmk images carry +movc-keeps-integer-source, which the 3 Nov compiler evaluation allowed only on a
  bitstream whose Q-04(b) reads PRESENT. Readings exist for bitstream 3 and for ObjBounds. None is recorded for C.5a or
  V1f. Every Capstone output passed specdiff on both. OPEN: the Q-04(b) state of C.5a and V1f.

## Addendum 34 (11 Nov 13:4x): P1's reading, the Plain task loses 6-13 % of the hart to 1-3 ms interruptions; and the exact probe, before it runs

**P1 reading** (board 1, V1f, kit bkit41-ids55-unify, out-tick-v1f-b1). 60.224 s at 25.000 MHz, 135 M loop steps.

| Gap size (cycles) | Count | Share of cycles |
|---|---|---|
| under 25,000 (the "small" class) | 488 (8.1/s), mean 8,076 cycles | 0.262 % |
| 25,000 and over (the "big" class) | 3,003, max 1,847,543 cycles | 18.841 % |

The histogram, in log2 buckets:

| Bucket | Count | Bucket | Count |
|---|---|---|---|
| 2^8 | 176 | 2^15 | 2,131 |
| 2^9 | 49 | 2^16 | 305 |
| 2^12 | 128 | 2^17 | 3 |
| 2^13 | 6 | 2^18 | 161 |
| 2^14 | 385 | 2^19 | 134 |
| | | 2^20 | 13 |

The transcript has 5 @@HB lines. In phase B, 4 gap passes (the rest fell outside the 16 KiB window), with a refill excess of
331 + 59 + 20 cycles per gap.

- **The 1 ms split was a wrong guess about the tick's cost.** The population of interest is 2^14 to 2^17 cycles
  (0.7-5 ms), about 2,824 events, about 47/s. Bounded by bucket edges, it takes **6.4 %-12.8 %** of the cycles, the size
  of the CPI residual (Addendum 28).
- The 2^18-2^20 gaps (10-80 ms) are the heartbeat's processes taking scheduler slices.
- F8's predicted "about 100/s" is NOT what was read: about 55/s, counting both classes. Whether this population is the
  periodic tick (with some ticks costing less than 400 cycles?) or another source is open. P1 kept no per-bucket sums and
  no timestamps, which makes it the wrong instrument to settle this.
- Refill after a gap is small for a 16 KiB working set (about 400 cycles). The steal dominates.

**P1b, before it runs.** perf/tick/hiccup2.c (executable 3ae5b31e206aad6a, stage perf/tick/stage-hiccup2) runs 60 s and
records up to 20,000 gaps with their start cycle and length. It prints:
- per-bucket counts and sums;
- the inter-arrival histogram of the 2^13-2^18 population, in 0.25 ms bins;
- the first 40 gaps raw.

It runs twice on board 1, V1f, after P2: A uses kit nohb2 and the nohb tools (no heartbeat), B uses the standard kit and
tools (driver perf/tick/tick2-v1f.sh).

**Predictions:**
- The 2^14-2^17 share is in [6.4, 12.8] % in both arms, and A has no 2^18+ gaps beyond a handful.
- If it is the periodic tick, the inter-arrival mode is 10 ms (HZ=100), or 20 ms if every other tick is cheap.
- Any other mode names a different source, which is then looked for in the monitor and kernel.

**Reading rule.** The measured A-arm share is the Plain-side interrupt cost, t. Under the reading that Capstone's masked
quanta avoid it, the tick-adjusted ratios are ratio / (1 − t). They are reported beside the measured ratios and never
instead of them. Whether to adjust is the lead's call, because a stock kernel on this board pays t.

**Dry run** (scratchpad tick2-dry): both arms reach the stage, and a fake pre-registration stops the run.

## Addendum 35 (11 Nov 14:1x): P1b read; the Plain task loses 9.4 % of the hart to the timer path, at 1.64 ms per event

**Where it ran.** It ran on board 2, not board 1: board 2 freed first, after gcc2. The driver is perf/tick/tick2b-v1f.sh,
which is tick2-v1f.sh plus `export BOARD_URL_FILE`. Board 1's queued copy was cancelled before it started, and the
cancellation is logged in tick2-v1f-b1.log. Board 2, V1f, out-tick2-v1f-b2.

| Arm | Wall | Gaps | 2^14-2^17 share | 2^18+ share | All gaps |
|---|---|---|---|---|---|
| A, no heartbeat | 60.026 s at 25.000 MHz | 3,453 | 9.384 % | 0 | 9.432 % |
| B, heartbeat | 60.124 s | 3,350 | 8.347 % | 10.594 % (313 gaps; 2 @@HB) | 18.991 % |

- **The population.** Arm A has 3,453 gaps, 57.5/s, with a mean of 40,991 cycles = 1.64 ms. The buckets are 2^14 574,
  2^15 2,420, 2^16 315 and 2^17 1.
- **Inter-arrival** (A): 20 ms in 2,699 events, 10 ms in 580, and a scatter of others. The source is periodic at the
  10 ms tick, and most events come every second tick. This is the timer path: no other periodic source runs at that
  rate. Why one tick in two stays under the 400-cycle threshold is not read here.
- **Prediction.** Addendum 34's [6.4, 12.8] % band holds: 9.38 % in A, 8.35 % in B.
- **Heartbeat.** B's 2^18+ share, 10.6 %, matches Addendum 33's measured f of 0.093-0.100. The two costs add, 9.4 +
  10.6 = 19.0.
- **F8 CONFIRMED as a cost on Plain.** t = 0.094 of the hart, spent away from the task at 1.64 ms per event. P1's refill
  excess is about 400 cycles per event, which is small beside that.
- **What is not read.** Which of the kernel, the M-mode monitor, SBI set_timer or trapped rdtime takes the 41,000 cycles
  is not read here. A stock kernel on this board pays the same cost.

**Tick-adjusted ratios, reported BESIDE the measured ones, never instead.** They are ratio / (1 − 0.094). This is an upper
bound for the adjustment: Capstone pays the tick outside its contexts. Outside-context time is 13.6 % of x264 train's run
and 23.6 % of lbm test's, so the true adjustment is 1-2 points smaller. Whether to adjust at all is the lead's call.

| | perl | gcc (module) | mcf | nab | xz c1 | imagick | lbm | x264 |
|---|---|---|---|---|---|---|---|---|
| Measured (heartbeat-corrected) | 1.238 | 1.196 | 1.180 | 1.088 | 0.999 | 0.977 | 0.926 | 0.882 |
| Tick-adjusted | 1.367 | 1.321 | 1.303 | 1.201 | 1.103 | 1.079 | 1.022 | 0.974 |

- **The "Capstone faster than Plain" readings.** After both corrections only x264 (0.974) stays below 1. imagick and lbm
  go above it.
- **gcc native train01.** 4,456.25 s, specdiff PASS. Against Plain-with-cache 3,280.9 s that is 1.358 raw and 1.497-1.509
  heartbeat-corrected. The reporting rule (Add. 23/27) reports native, with the module route's 1.196 beside it.
- **Same-environment test-size pairs** (V1f, neither arm with the heartbeat):

  | Benchmark | Native | Plain | Ratio |
  |---|---|---|---|
  | imagick | 20.59 s | 16.13 s | 1.277 |
  | nab | 894.17 s | 775.94 s | 1.152 |
  | lbm | 1,658.05 s | 1,713.29 s | 0.968 |

  Native comes from Addendum 21's run 4 (board 2), Plain from Addendum 30 (board 1). This lbm ratio is consistent with
  the tick cost.

## Addendum 36 (11 Nov 15:2x): Addendum 31's RVC pair read, RVC ruled out (F7); and P3, hiccup2 inside a Capstone context, before it runs

**RVC pair** (board 3, C.5a, kit bkit27-gtq, out-rvc-b3; flashed C.5a 12:46, chain 12:46-15:08, bitstream 3 flashed back
at 15:10 and read back). Each arm ran once, alternating RVC then no-RVC per benchmark. Every arm is rc 0 with specdiff
PASS.

| Benchmark | RVC (plain-rv64r) | No RVC (rv64g) | No RVC / RVC | Hart CPI RVC / no RVC | C.5a record |
|---|---|---|---|---|---|
| imagick | 19.55 s | 19.37 s | 0.991 | 4.10 / 4.07 | 19.46 s |
| nab | 862.23 s | 867.03 s | 1.006 | 3.15 / 3.17 | 862.99 s |
| lbm | 1,892.96 s | 1,890.02 s | 0.998 | 6.15 / 6.15 | 1,895.0 s |

- **R2b holds.** Every pair is within ±1 %, inside the ±2 % band, and the RVC arms reproduce the record within 0.5 %. R2a
  fails: neither lbm nor nab runs ≥ 5 % faster without RVC.
- **R1, partly.** rc and specdiff hold. The QEMU instruction counts are within 0.5 % for lbm (−0.24 %) and nab (−0.10 %),
  and not for imagick (+0.86 %, 0.1175 G against 0.1165 G). The board's hart_instret for imagick goes the other way
  (−0.29 %), and that count includes the kernel.
- **F7 RULED OUT.** RVC does not explain Plain's higher CPI. The rtl-oracle prediction holds: on this core RVC costs one
  cycle only on a redirect to a 2-mod-4 target, which is negligible here. Plain-relative ratios carry no RVC correction.

**P2 arm A** (board 1, C.5a, no heartbeat): Capstone lbm test 1,651.3 s (spatial) and 1,651.1 s (lifetimes), specdiff
PASS, 0 @@HB. Arm B (heartbeat) is running. Its reading comes in the next addendum.

**P3: what a Capstone task loses to exits and interrupts.** Addendum 35 measured Plain's loss as t = 0.094 and bounded the
tick adjustment from above, because Capstone pays ticks outside its contexts. P3 measures the Capstone side directly.
- **The probe.** perf/tick/hiccup2.c, unchanged, built with each module-route SDK (spec/sdks-dev2mk/sdk-{spatial,lifetimes}
  capstone-cc -O2). spatial is 193c49e5f64c0278, lifetimes 82c05ac3c35e938e.
- **The stage.** perf/tick/cap/stage-cap-hiccup2 is a copy of stages-eval-boarddevmk/538.imagick_r-test. Only
  imagick_r_base.capstone-{spatial,lifetimes} and their plan.json hashes changed. The validation step fails by
  construction, because the probe writes no image. The probe's lines are read from files/<arm>/test_convert.out.
- **The run.** Board 1, C.5a, kit bkit23-all2-b1-nohb with the no-heartbeat tools (P2's arm A environment). Driver
  perf/tick/caphic.sh (from capnohb.sh, chain A only). It waits for capnohb's driver to exit. Dry run is clean, and the
  extraction line was checked on Addendum 35's Plain transcript.
- **What it reads.** Each gap is a step over 400 cycles of the free-running cycle counter inside the context. A gap is
  any time the hart spends away from the task: a context exit (quantum, syscall, fault) and whatever the kernel and
  monitor do before re-entry. The probe's own clock_gettime every 2^20 iterations is a syscall, so it adds a few hundred
  exits of its own. The gap count and IA histogram show them.
- **Predictions:**
  - **P3a, ticks masked in context** (the working assumption of Addendum 35): HIC2-ALL ≤ 3 %, and no 10/20 ms
    inter-arrival peak.
  - **P3b, ticks reach the context:** HIC2-ALL ≥ 8 %, with the 10/20 ms peak.
  - **Otherwise:** the share is reported as read, and it replaces the 0 that the upper bound assumed for Capstone.
  - **Fault case:** if rdcycle traps in a context, the arm ends at once with a non-zero rc and nothing is read.
- **How it is used.** The tick-adjusted column becomes ratio × (1 − t_C) / (1 − t_P), with t_P = 0.094 and t_C read
  here. It is still reported beside the measured ratio, never instead.

**Operational note.** Building P3's stage, a `cp -a` of stages-eval-b1/538.imagick_r-test-devmk copied the SYMLINK, and
writes into the "copy" landed in stages-eval-boarddevmk/538.imagick_r-test from 15:13 to 15:15. Both imagick
executables were restored from tree-boarddevmk's exe directory (b4073795…, f50dc465…). plan.json was restored with
the same writer format as make-stage.py, and all four of its hashes verify. No run read that stage in the window. The P3
stage is now a real copy: no symlinks, and no file with more than one link.

## Addendum 37 (11 Nov 15:5x, before it runs): two board checks of the tick hypothesis on SPEC test, boards 2 and 3

**What the existing counters already say.** Plain lbm test retires 7.70 G instructions on the hart (out-rvc-b3), while
QEMU counts 6.66 G for the program itself. The excess, about 1.0 G over 1,893 s, is kernel and monitor work. At 57.5
expensive events per second (Addendum 35), that is about 9,600 instructions per event, if most of the excess is the
tick. nab gives about 0.60 G over 862 s. Both include syscalls and page faults, so this is an upper bound per event. A
stock OpenSBI timer path is a few hundred instructions, so the cost may belong to this platform's firmware and not to
"any kernel on this board", as Addendum 35 said. That sentence is withdrawn until P4 reads.

**P4, tickcost on board 2** (V1f, kit nohb2, nohb tools, chain A only; driver perf/tick/tick3-v1f.sh, stage
perf/tick/stage-tickcost, probe perf/tick/tickcost.c, Plain build 11543a68ab1376a4). The probe is in place of imagick_r,
and the validation fails by construction. It measures four things:
- rdtime cycles (median of 10,000);
- getppid cycles and instructions (median of 10,000);
- 30 s of a rdcycle/rdinstret loop, with the cycles and the instructions retired inside every gap over 400 cycles;
- the same 30 s while streaming over 8 MiB.

Predictions:
- **P4a.** rdtime ≥ 500 cycles means the time CSR is trapped and emulated in M-mode; ≤ 50 means it is native.
- **P4b.** The mean instructions inside a ≥ 2^13-cycle gap are ≥ 2,000, so the event is software-heavy. This would be
  consistent with the ~9,600 above. If they read < 500, the 41k cycles are stalls, not code.
- **P4c.** The idle loop's big-gap share reproduces Addendum 35's 9.4 % within ±1.5 points.
- **P4d.** If the streaming loop's big-gap share exceeds the idle one by ≥ 1 point, a tick costs memory-bound code
  (lbm, x264) more than 9.4 %. If not, the 9.4 % stands for them.
- If rdinstret is refused (SIGILL), the probe prints TC-NOTE and the instruction columns read 0. Then P4b is not read.

**P5, same-kit pairs on board 3.** Flash C.5a, run perf/samekit/b3.list, then flash bitstream 3 back. Driver
perf/samekit/samekit.sh is from perf/b3/gt-run-ga2.sh. Both arms run on ONE kit, bkit23-all2-b1-nohb, the kit of every
module-route record and of P2's arm A, with the no-heartbeat tools copy. Only the stage differs between the Plain arm
and the Capstone arm.

| # | Stage | Arm |
|---|---|---|
| 1 | plainreuse 538.imagick_r-test-plain-rv64r | Plain |
| 2 | stages-eval-b1 538.imagick_r-test-devmk | Capstone spatial, lifetimes |
| 3 | stage-tickcost | P4 repeated on C.5a |
| 4 | plainreuse 544.nab_r-test-plain-rv64r | Plain |
| 5 | stages-eval-b1 544.nab_r-test-devmk | Capstone |
| 6 | plainreuse 519.lbm_r-test-plain-rv64r | Plain; Capstone lbm on this kit is P2 arm A, 1,651.3 s |

- **board-env-check C2 is waived.** The kit module is the 16-bit-table default build, and C.5a is an 18-bit-table
  bitstream. This is the same kit every module-route record used. Any other failing check stops the run. The waiver
  logic was negative-tested.
- **x264 is not in the set.** Its Plain test takes 50,441 s, which does not fit before the deadline.
- **Predictions:**
  - **P5a.** The Plain arms reproduce the V1f no-heartbeat Plain times within ±2 %: imagick 16.13 s, nab 775.94 s,
    lbm 1,713.29 s. This is F2, the bitstream and kit, for Plain.
  - **P5b.** The same-kit ratio Capstone / Plain equals the heartbeat-corrected one within ±3 %. For lbm that predicts
    1,651.3 / 1,713 ≈ 0.964.
  - **Reporting.** The same-kit ratios are reported as the MEASURED ratios with the fewest confounds. The tick
    adjustment is applied beside them, never instead, with P3 and P4 as inputs.

## Addendum 38 (11 Nov 16:1x): P2 read (F9 false); the table re-derived from its sources; the two gcc numbers are not one variable

**P2 arm B** (board 1, C.5a, kit bkit23-all2-b1, heartbeat): Capstone lbm test 1,810.12 s (spatial) and 1,810.91 s
(lifetimes), specdiff PASS. It reproduces the record of 1,809.88 s to 0.01 %. There are 104 @@HB lines over the two
arms, about 52 per arm.
- The Capstone heartbeat share is f_C = 0.0877 (spatial) and 0.0882 (lifetimes), against Plain lbm's 0.0929.
- The cost per beat is (1,810.12 − 1,651.3) / 52 = 3.05 s, inside Plain's 3.0-3.8 s band. **F9 FALSE:** the Capstone
  arm pays the heartbeat as Plain does.
- The residual on module-route ratios is × (1 − f_C) / (1 − f_P) = × 1.006. It is measured on lbm only and is reported
  as a range, not applied as a point.

**Table re-derived from the recorded times** (no deviation either way):
- **perlbench, 1.238.** Raw 54,021.76 / 48,206.54 = 1.1206, divided by (1 − perl's own f 0.0946).
- **mcf, 1.176** (it was 1.180, the top of its range). Raw 47,980.64 / 45,160.79 = 1.0624. mcf has no f of its own, so
  the mean of lbm's and nab's f, 0.0966, is used.
- **imagick, 0.973** (it was 0.977, the top of its range). Raw 38,251.91 / 43,517.91 = 0.8790, with the same mean f.
- **lbm, 0.926.** Raw 20,773.27 / 24,724.69 = 0.8402, with lbm's own f.
- **Module route, both arms with the heartbeat:**
  - nab 79,730.08 / 73,311.84 = 1.088;
  - x264 run0 36,355.01 / 41,232.13 = 0.882;
  - xz c1 8,730.5 / 8,735.95 = 0.999;
  - gcc 3,924.91 / 3,280.9 = 1.196.

**gcc: the native and module numbers differ in the EXECUTABLE, not only the route.**
- **Module, 3,924.91 s,** is the dcnofp image 736f881f4958ac3b (results/2026-11-08-dcnofp), on kit bkit32-q32 with
  a 32 M-cycle quantum. It is built with llvm-c4, `-mllvm -capstone-direct-calls` (default off, on a compiler-lane
  branch) and `-fomit-frame-pointer`. Its in-context instructions are 6.3 % fewer than the c3f build's.
- **Native, 4,456.25 s,** is df93c433bb1a5418: the 3 Nov devmk make lines, llvm-c1, with the frame pointer and without
  direct calls. It was replayed with the native SDK
  (docs/history/09-11-2026_17-10-00_native-quick-test-e5native.md).
- **The other Capstone numbers are mostly devmk builds.** perl, mcf, imagick, lbm and nab are devmk builds through
  each route. x264 is c3f.
- **Conclusions:**
  - gcc's module 1.196 used a compiler configuration no other benchmark got. That is favourable to Capstone.
  - gcc's native 1.503 is not a native-route effect measured on the same code. At test size, the same devmk exe ran
    native 20.17 s against the module's 25.38 s in-VM (Addendum 21).
  - The like-for-like module number on the c3f exe is 4,214.98 s (out-q32-b3, same kit) = 1.285.
  - Which gcc build is the reported one is the lead's call. The reporting rule's "native" assumed the same exe, and it
    does not hold here.

## Addendum 39 (11 Nov 16:1x): P4 and P3 read; and P3b (the q32 kit) and P6 (the native route), before they run

**P4 read** (board 2, V1f, kit nohb2, out-tick3-v1f-b2; tickcost a09d… rebuilt as 11543a68ab1376a4, specdiff fails by
construction).

| Line | Reading | Prediction |
|---|---|---|
| rdtime | median 1,116 cycles; a rdcycle pair takes 4 | P4a holds: the time CSR is trapped and emulated in M-mode |
| getppid | median 1,891 cycles, 275 instructions | — |
| Idle, 30.5 s | 1,655 gaps ≥ 2^13; mean 42,554 cycles, of which 4,645 instructions; share 9.233 % (all gaps 9.262 %) | P4b holds (≥ 2,000 instructions). P4c holds (9.43 % ± 1.5) |
| Streaming over 8 MiB, 31.8 s | 3,178 gaps ≥ 2^13; mean 27,262 cycles, of which 3,716 instructions; share 10.906 % | P4d holds: +1.67 points ≥ 1 |

- **Software, not stalls.** The interruption is about 4,600 instructions at a CPI of about 9 (cold caches, 16 B lines),
  plus a trapped rdtime at 1,116 cycles each. That is about what a stock Linux tick plus SBI costs in instructions.
  The cycles come from this FPGA system's CPI.
- **Memory-bound code pays more.** Under the streaming load every tick shows: 100/s, against 54/s idle. The share is
  10.9 % rather than 9.2 %.

**P3 read** (board 1, C.5a, kit bkit23-all2-b1-nohb, out-caphic-b1). hiccup2 inside a Capstone context, both arms. The
validation fails by construction.

| Arm | Wall | Gaps | 2^17 | 2^18 | 2^19 | All |
|---|---|---|---|---|---|---|
| spatial | 60.166 s | 297 | 148, 1.784 % | 144, 4.532 % | 5, 0.180 % | 6.496 % |
| lifetimes | 60.132 s | 297 | 148, 1.765 % | 146, 4.571 % | 3, 0.107 % | 6.443 % |

- **Timeline** (raw gaps). Every 326-328 ms there is a ~18-22 ms gap: the 8,000,000-cycle supervisor quantum, plus the
  exit and re-entry. 76-80 ms after each comes a ~6.5-8.5 ms gap. The module's step_budget_ns is 80,000,000.
- **There is no 10/20 ms population.** Neither P3a (≤ 3 %) nor P3b (≥ 8 % with the tick peak) holds. The share is
  read: **t_C = 0.065** on this kit, in the idle loop.
- **Capstone pays for periodic preemption too**, batched at the quantum, at about 69 % of Plain's 9.4 %. So
  Addendum 35's tick-adjusted column (t_C = 0) overstated the adjustment. The symmetric adjustment, removing
  periodic-interrupt cost from both sides, is ratio × (1 − t_C) / (1 − t_P):
  - × 1.031 for t_P = 0.093 (idle);
  - up to × 1.049 for memory-bound code (t_P = 0.109), if t_C does not also grow under memory load. That is not
    measured.

**Still to read before t_C applies to the train numbers.** They ran on other configurations:
- **P3b.** The module train runs of gcc, nab and x264 used kit bkit32-q32: a 32,000,000-cycle quantum and the 18-bit
  module. A new kit, t3-smoke/bkit32-q32-nohb, is bkit32-q32 with only the runner idle limit = step budget and the
  watchdog entry-stall abort off. A board.env comment says so, and SHA256SUMS is regenerated. Both kits verify, and the
  diff is checked in the driver.
  - Driver perf/tick/caphic-q32.sh (caphic.sh with that kit, module 65ede944a87ea742 and vexec 1d8b4f0829bef0e7);
    board 1; the same stage-cap-hiccup2. Its dry run is clean.
  - Prediction: a quantum gap every ~1.3 s. The share is in [1.5, 7] %: low if the exit's fixed cost dominates, about
    6.5 % if the cost is per deferred tick. It is reported as read.
- **P6.** The native train runs of perlbench, mcf, imagick, lbm and gcc used the native route. hiccup2 is built with
  native01/sdk-native-c1 (68867466…). Stage perf/tick/nt/qt-hiccup is a real copy of native01/board-stages/qt-imagick.
  - Only two things change: the imagick executable, and one run.sh line that copies the probe's HIC2 lines to the
    console. qt-hiccup.SHA256SUMS is regenerated; the file set is the same, and only those two hashes differ.
  - The driver is perf/train/nat-v1f.sh (step nat:hic, kit bkit41-native-v1f-mcf, 1,800 s, fallback none), on board 2,
    V1f. Its dry run is clean.
  - The validation fails by construction, so the step logs NAT FAIL. That is expected and is not a route failure.
  - Prediction: if a native task takes ticks as a Linux task does, the share is 9.2 % ± 1.5 with the 10/20 ms peak,
    and the native-route ratios need no tick adjustment. If the native route masks ticks, it is read like P3.

## Addendum 40 (11 Nov 16:3x): P3b and P6 read; the tick-neutral table; the lead's decisions on gcc, native builds and the baseline

**P3b read** (board 1, C.5a, kit bkit32-q32-nohb, out-caphic-q32-b1). In both arms there is one gap of 23-27 ms every
~402 ms, 149 of them, all 2^19. Shares are 5.909 % (spatial) and 5.935 % (lifetimes), so **t_C(q32) = 0.059**, inside
the predicted [1.5, 7] %. The period is ~402 ms, not the 1.28 s quantum, so another limit sets the exit rate. That limit
is not identified here.

**P6 read** (board 2, V1f, native route, out perf/tick/nt/out/v1f-hic-b2). The run is rc 0. The validation is rc 255,
which was expected, so the driver logged NAT FAIL by construction. There are 298 gaps: 148 of 2^17 (2.010 %), 149 of
2^18 (4.540 %) and one of 2^16. The pattern matches P3's. **t_C(native) = 0.0655.** The native route also batches
preemption at about 6.5 %, and the native prediction "9.2 % with the 10/20 ms peak" fails: no 10/20 ms population.

**Tick-neutral factors**, applied symmetrically: × (1 − t_C) / (1 − t_P), with t_P = 0.0933 (the mean of Addendum 35's
9.432 % and P4's 9.233 %, idle) and t_C read on each ratio's own configuration, also idle. A memory-heavy load raised
Plain's share to 10.9 % (P4d). Capstone's share under memory load is not measured, so idle against idle is the
unbiased point. The upper column assumes Capstone's share does not grow.

| Benchmark | Measured | Configuration, t_C | Factor | Tick-neutral | Upper |
|---|---|---|---|---|---|
| perlbench | 1.238 | native, 0.0655 | 1.0307 | 1.276 | 1.298 |
| gcc | 1.196 | module q32, 0.059 | 1.0376 | 1.241 | 1.263 |
| mcf | 1.176 | native | 1.0307 | 1.212 | 1.233 |
| nab | 1.088 | module q32 | 1.0376 | 1.129 | 1.149 |
| xz c1 | 0.999 | module 8 M (nset), 0.065 | 1.0315 | 1.030 | 1.049 |
| imagick | 0.973 | native | 1.0307 | 1.003 | 1.020 |
| lbm | 0.926 | native | 1.0307 | 0.954 | 0.971 |
| x264 | 0.882 | module q32 | 1.0376 | 0.915 | 0.931 |
| geomean | 1.052 | | | 1.087 | |

The module-route heartbeat residual (× 1.006, P2) is not applied. lbm and x264 stay below 1 after this correction.
Their residual is still open.

**Lead's decisions (11 Nov, this conversation):**
- gcc train reports the module route, 1.196 (dcnofp, 3,924.91 s). That replaces Addendum 38's reporting-rule reading.
- From now on, native-route executables are built with the best optimised compiler configuration available.
- **The baseline is "ordinary RISC-V".** Plain is measured on a bitstream of the same SoC with the Capstone extension
  off: the Capstone config with only CAPSTONE_EXT = 0 and without CAPSTONE_ITFC_EN, with caches, TLBs and predictors
  unchanged. The firmware is stock OpenSBI and the Linux kernel is the same.
  - **CORRECTION.** The 2026-11-01 pre-registration called ObjLife / Plain "the cost of Capstone as a whole". It is the
    cost of Capstone's software on Capstone hardware and firmware. Any cost the extension's logic or the Capstone
    monitor imposes on ordinary code is paid by Plain too, and cancels. Every Plain-relative ratio so far is that
    narrower quantity.
  - The baseline build and its firmware bring-up are being scoped now. The baseline is not available before the
    20:00 deadline.

## Addendum 41 (11 Nov 16:4x, before it runs): Plain on stock OpenSBI, same bitstream (plan step A4), board 2

**Why.** The lead set the baseline to ordinary RISC-V (Addendum 40). The first separable part is the firmware: Plain on
the Capstone monitor against Plain on STOCK OpenSBI, with the bitstream, kernel, initramfs base and device tree held
fixed.
- **Stock firmware.** stock-sbi d6b78e4: upstream OpenSBI v1.3 (the opensbi repo's `vanilla` tag, 057eb10) plus the
  board's UART clock, 25 MHz / 57600, which the fork and the device tree use. It is a branch stock-ariane-25mhz in a new
  worktree.
- **Recipe.** perf/stockfw/build-stock-fw.sh has the same interface as build-fpga-fw.sh and builds no monitor. Its gate
  checks that there are no Capstone monitor symbols, that the Image is in the payload, and that the platform name is
  the stock one.
  - A test build passed: 17,467,912 B.
  - The symbol check fires on a Capstone firmware ELF (4 symbols).
- **Kit t3-smoke/bkit41-stockfw-v1f.** It is bkit41-ids55-unify-nohb2 with these changes, nothing else:
  - the firmware lines: recipe, wrap = mon = stock-sbi, commit d6b78e4, a placeholder define;
  - BOARD_RUNGS="vbatch", since the b0 rungs are Capstone domains;
  - no debug-aperture reads.
  
  SHA256SUMS is regenerated, and the driver checks the diff (12 board.env lines and the aperture line).
- **Runner perf/stockfw/run-stage-noctl.py.** It is the nohb tools' run-stage.py with spec-board's `--control` dropped.
  The known-good control is a Capstone image and cannot run without the monitor, so these boots have no Capstone
  control. A boot that fails to come up yields no stage result.
- **Driver perf/stockfw/stockfw-v1f.sh** (from tick3-v1f.sh), board 2, V1f. List: tickcost, hiccup2, then Plain
  imagick, nab and lbm test (plainreuse rv64r). Its dry run is clean, and the extraction line was checked on a P1b
  transcript.
- **Predictions:**
  - **S1.** It boots and every Plain stage passes specdiff, except the two probes, whose validation fails by
    construction.
  - **S2.** The tick share falls if the Capstone monitor inflates it. That means hiccup2 < 9.4 % and tickcost's mean
    gap < 42,554 cycles. rdtime < 1,116 cycles if stock emulation is cheaper.
  - **S3.** Plain test times change by about the tick difference: × (1 − t_stock) / (1 − t_monitor). This is
    against Addendum 30's V1f no-heartbeat Plain: imagick 16.13 s, nab 775.94 s, lbm 1,713.29 s.
  - Whatever the reading, it is the firmware's cost on ordinary code.

**RTL finding for step A1 (rtl-oracle, verilator lint of a V1f copy; the repo was not touched).** The Capstone fork
with CAPSTONE_EXT = 0 and no CAPSTONE_ITFC_EN does NOT elaborate:
- ex_stage.sv:2161-2172 has a syntax error;
- after that is patched in the copy, 166+ "can't find definition" errors remain, from uses outside `ifdef
  CAPSTONE_ITFC_EN`.

Ungated Capstone glue also stays on the next-PC path (frontend.sv:458-466) and adds a regfile read port
(issue_read_operands.sv:1662). So "the fork with the flag off" is not an ordinary core without RTL work. The
alternative is upstream CVA6 at the fork's base, with the Capstone config's parameters. That is being scoped.

## Addendum 42 (11 Nov 17:0x, before it runs): plan track B2, the Capstone compiler chosen on the board; the baseline bitstream in synthesis

**B1 so far (QEMU, test size; perf/ccsel, subagent build, verified by exact rebuild of every base exe).** Every
candidate is rc 0 with specdiff PASS, and specdiff's positive control fires (one digit changed → 0/1, rc 1). In-context
instret against base:

| Benchmark | dcnofp | dcnofprs |
|---|---|---|
| imagick | −0.60 % | −0.62 % |
| nab | −2.19 % | −2.44 % |
| lbm | −1.70 % | −1.70 % |

- **gcc.** The c3f recipe exes read −14.9 % (c3f), −18.3 % (c3f+dcnofp) and −18.8 % (c3f+dcnofprs). This is a recipe
  difference, not the flags alone.
- **Still in QEMU:** mcf, x264, perl and xz.
- **The candidates:**
  - dcnofp = gate-dc SDK (llvm-c4 98337e20) with `-capstone-direct-calls -fomit-frame-pointer`;
  - dcnofprs = dcnofp plus `-capstone-drop-redundant-shrink`.
- **The `+movc-keeps-integer-source` question (F11) is unchanged.** Both SDKs carry it, and so does every base.

**B2, board 1** (C.5a, kit bkit23-all2-b1-nohb, nohb tools, one-arm lifetimes stages from perf/ccsel/stages). The
list alternates base and dcnofprs: imagick, nab, lbm (perf/ccsel/b2.list). The driver is perf/ccsel/ccsel-b2.sh, from
caphic.sh, and its dry run is clean.
- **Prediction:** dcnofprs is faster by about the instret change, nab −2 to −3 % and lbm −1 to −2 %. imagick is within
  noise.
- **Rule:** a benchmark takes dcnofprs if it is not slower than base on the board.

**Track A.** The baseline RTL is upstream CVA6 2ef1c1b1f (the fork point), plus:
- the Capstone board's clock and pins;
- `baseline_cv64a6_imafdc_sv39` = the Capstone config minus CAPSTONE_ITFC_EN and CAPSTONE_EXT;
- the five stock-logic fixes the fork carries: 213378a11, the c77c65324 MMU hunk, 6cbdaeeb4, cc1159a51, and
  a73a7c670 + e7af48457;
- the zero-initialised FPGA regfile.

Two fork commits that fix bugs only the fork has (546807884, ae21e121e) are not ported. The result is capstone-ariane
branch riscv-baseline-fixes d31345cf3. Verilator lint gives 0 errors, the same as stock cv64a6. The guarded full
synthesis has run on apollo since 16:50. A pure-upstream build (riscv-baseline fbbcf282e) was started first and
stopped at 15 min, because the memory projection allows one job.

## Addendum 43 (11 Nov 17:2x): stock OpenSBI on V1f, read in part; the ordinary RISC-V baseline bitstream sealed; phase 1 on it, before it runs

**Addendum 41, read so far** (board 2, V1f, kit bkit41-stockfw-v1f, out-stockfw-v1f-b2). **S1 holds:** stock OpenSBI
v1.3 boots this SoC and Linux, the module loads, and imagick and nab pass specdiff. lbm is still running.

| | Capstone monitor (V1f) | Stock OpenSBI v1.3 (V1f) |
|---|---|---|
| hiccup2 loss | 9.432 % | 12.617 % (6,163 gaps; 2^14 5,293, 10.55 %; inter-arrival 10 ms: 2,993) |
| tickcost idle / streaming | 9.233 % / 10.906 % | 12.423 % / 13.724 % |
| mean big gap, idle | 42,554 cycles, 4,645 insns, 54/s | 31,057 cycles, 3,339 insns, 100/s |
| rdtime | 1,116 cycles | 2,231 cycles |
| Plain imagick / nab test | 16.13 s / 775.94 s | 16.89 s / 802.37 s |

- **S2 fails, in the other direction.** Stock OpenSBI costs Plain MORE than the Capstone monitor does. Every tick is
  taken at 100/s and each is cheaper, but the total is +3.2 points. rdtime emulation costs twice as much.
- **S3 holds in sign:** imagick +4.7 %, nab +3.4 %.
- **What it means.** Under the Capstone monitor Linux runs through the monitor's own trap path (the fork's interrupt
  handling). That path is not stock OpenSBI with tuning. An "ordinary RISC-V" baseline should still not use a firmware
  that is slower than necessary, so the baseline's firmware is chosen by measurement among stock releases. It is the
  fastest of v1.3, v1.6 and v1.9, which is the choice least favourable to Capstone.

**The ordinary RISC-V bitstream** (Addendum 42's RTL, riscv-baseline-fixes d31345cf3). The full flow ran on apollo
under vivado-guard in 27 min 39 s, peak RSS 16.22 GB.
- **Clean:** routing errors 0, timing loops 0, Synth 8-326 0, LUTLP-1 0.
- **Timing:** the core clock is 25 MHz (40 ns), the same as the Capstone bitstreams, with WNS +13.228 ns. All timing
  is met.
- **Sealed:** caplifive_rvbase_d31345cf3.bit, sha256 51e883b7c0beffa23ba012ec9cdef8d32c7129ba15c83ce1402b090395e5f8ed.
  Reports and SHA256SUMS are in synth/rvbasefix-d31345cf3-full-out/seal.

**Phase 1** (board 2, after Addendum 41's run). Driver perf/stockfw/rvbase-b2.sh: flash the baseline (sha checked,
read back), then tickcost and hiccup2 under three stock firmwares, one chain each:
- kit bkit42-rvbase-sbi13: OpenSBI v1.3, d6b78e4;
- kit bkit42-rvbase-sbi16: v1.6 + the board UART, 121a885;
- kit bkit42-rvbase-sbi19: v1.9, platform/generic from the device tree.

Each kit is bkit41-stockfw-v1f with only BOARD_BITSTREAM and the firmware lines changed, and the driver checks this.
The dry run is clean.
- **Predictions:**
  - The baseline boots Linux on stock OpenSBI.
  - getppid is within ±10 % of 1,891 cycles: same core pipeline, different extension.
  - Each firmware's tick share is read. The phase-2 firmware is the one with the lowest hiccup2 share.
  - If the baseline does not boot, nothing is read about the firmwares, and the bitstream goes back to RTL.

## Addendum 44 (11 Nov 19:2x): P5 and phase 1 read; OpenSBI v1.3 chosen for the baseline; B2 in part; phase 2 (Plain SPEC on the ordinary RISC-V baseline), before it runs

**P5, same kit, read** (board 3, C.5a, kit bkit23-all2-b1-nohb, out-samekit-b3). Every arm passes specdiff.

| | Plain | Capstone lifetimes | Same-kit ratio |
|---|---|---|---|
| imagick test | 16.10 s | 19.14 s (spatial 19.18 s) | 1.189 |
| nab test | 776.27 s | 903.36 s (spatial 899.50 s) | 1.164 |
| lbm test | 1,713.76 s | 1,651.3 s (P2 arm A) | 0.964 |

- **P5a holds.** Plain reproduces the V1f no-heartbeat times: 16.10 vs 16.13 s, 776.27 vs 775.94 s, 1,713.76 vs
  1,713.29 s. For Plain, F2 (the bitstream and kit) is ruled out between C.5a and V1f.
- **P5b holds for lbm:** 0.964 against the predicted 0.964.
- tickcost on C.5a reads 9.28 % idle, inside P4's V1f range.
- **Scope.** All of these ratios share one SoC: both arms ran on the Capstone hardware and the Capstone monitor. Phase 2
  replaces the Plain side.

**Addendum 41, lbm:** stock OpenSBI v1.3 on V1f gives 1,774.28 s, specdiff PASS. This is +3.6 % on the Capstone
monitor's 1,713.29 s, the same sign as imagick (+4.7 %) and nab (+3.4 %).

**Phase 1, read** (board 2, baseline caplifive_rvbase_d31345cf3, out-rvbase-tick-b2).

| | Capstone SoC (V1f), stock v1.3 | Baseline, stock v1.3 | Baseline, stock v1.6 | Baseline, stock v1.9 |
|---|---|---|---|---|
| boots | yes | yes | yes | **no** |
| hiccup2 loss | 12.617 % | 7.764 % | 8.530 % | — |
| tickcost idle / streaming | 12.423 / 13.724 % | 7.604 / 8.136 % | 8.390 / 9.007 % | — |
| mean big gap, idle (cycles / insns) | 31,057 / 3,339 | 19,012 / 3,422 | 20,974 / 3,794 | — |
| rdtime | 2,231 cycles | 1,200 | 1,501 | — |
| getppid | 1,891 cycles, 275 insns | 1,018 cycles, 275 insns | 1,018 cycles | — |
| STREAM steps/s | 319,491 | 456,044 | 451,678 | — |

- **The baseline boots Linux on stock OpenSBI.** The module loads, and the Plain probes run.
- **The getppid prediction fails, by a wide margin.** The baseline needs 1,018 cycles, not 1,891 ±10 %. Both SoCs run
  the same 275 instructions under the same firmware. So the Capstone SoC runs this ordinary kernel path at 1.86× the
  cycles per instruction. The tick handler shows the same effect: about the same instructions, 19,012 against 31,057
  cycles. Streaming over 8 MiB is 43 % faster on the baseline.
  - The baseline is upstream CVA6 2ef1c1b1f. That commit is the merge-base of the Capstone fork's branches
    (capstone-bootstrap, v1f-ids55-native) with upstream, so both bitstreams start from the same CVA6.
  - The difference is therefore what the Capstone fork changed in the core and the SoC. Which of those changes costs
    this is not resolved here.
  - **Consequence.** Every ratio against Plain on the Capstone SoC, the P5 table included, understates Capstone's
    cost relative to ordinary RISC-V.
- **v1.9 does not boot on this SoC.** The bootrom prints its banner, the JTAG load completes, and then the console is
  silent until the 1,800 s login timeout. The likely cause is that platform/generic needs a device tree pointer in a1,
  which the board flow does not pass. This is not debugged here. The choice is between v1.3 and v1.6, and the newer
  release is not the faster one.
- **Choice: OpenSBI v1.3, kit bkit42-rvbase-sbi13.** It has the lowest hiccup2 share, 7.764 %, as pre-registered.

**B2, read in part** (board 1, C.5a, kit bkit23-all2-b1-nohb, no heartbeat; Capstone lifetimes). Every arm passes
specdiff.

| | base (devmk, llvm-c1) | dcnofprs (c4, direct calls, no frame pointer, drop redundant shrink) | dcnofprs / base |
|---|---|---|---|
| imagick test | 19.16 s | 19.09 s | 0.996 |
| nab test | 903.88 s | 871.88 s | 0.965 |
| lbm test | 1,652.37 s | running | |

The base arms reproduce P5's lifetimes arms (19.14 s, 903.36 s) and P2 arm A (1,651.3 s) to 0.1 %.

**Phase 2: Plain SPEC on the ordinary RISC-V baseline.** Driver perf/stockfw/rvbase-p2.sh: flash the baseline (sealed
sha checked, read back), then a chain over plainreuse Plain stages with kit bkit42-rvbase-sbi13. The kit has no
heartbeat and no Capstone control step. These are the same stages and executables as every Plain record so far.

| Board | List | Stages |
|---|---|---|
| 3 | rvbase-p2c.list | imagick test, nab test, gcc train01 |
| 2, after phase 1 | rvbase-p2d.list | lbm test |

The dry runs of both reach every stage.
- **Cross-board.** P5a showed Plain does not depend on the bitstream and kit between C.5a and V1f. Boards 2 and 3 are
  the same hardware, so a stage is read on whichever board runs it.
- **Predictions:**
  - Every stage passes specdiff.
  - Each test stage is faster than on the Capstone SoC with the same stock v1.3 firmware (imagick 16.89 s, nab
    802.37 s, lbm 1,774.28 s).
  - The ratio of baseline time to V1f-stock time lies in [0.70, 0.96]:
    - the lower bound is the STREAM ratio, 319,491 / 456,044;
    - the upper bound is the tick share alone, 12.6 % to 7.8 %.
  - gcc train01 is below the lowest Plain record on the Capstone SoC, 3,280.89 s.
- **Reporting.** Capstone / ordinary RISC-V for a benchmark is:
  - Capstone, the best build chosen in B2, on the Capstone SoC, divided by
  - Plain on the baseline.

  The P5 same-kit ratio is reported beside it. That ratio holds the SoC fixed, so it isolates the software cost.
