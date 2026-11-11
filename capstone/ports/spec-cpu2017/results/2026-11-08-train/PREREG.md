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
