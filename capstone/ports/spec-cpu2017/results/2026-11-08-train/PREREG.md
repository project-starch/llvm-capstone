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

## Addendum 45 (11 Nov 19:5x): phase 2 test stages and B2 read. Against ordinary RISC-V, Capstone costs 1.31-1.71x on SPEC test, and most of that is the Capstone SoC running ordinary code slower

**Phase 2, read** (OpenSBI v1.3, kit bkit42-rvbase-sbi13, no heartbeat). Every stage passes specdiff, and the
executables are the plainreuse ones of every Plain record.

| | Plain, baseline | Plain, Capstone SoC (V1f), stock v1.3 | baseline / V1f-stock | Plain, Capstone SoC, Capstone monitor (P5) |
|---|---|---|---|---|
| imagick test (board 3) | 11.17 s | 16.89 s | 0.661 | 16.10 s |
| nab test (board 3) | 664.14 s | 802.37 s | 0.828 | 776.27 s |
| lbm test (board 2) | 1,098.43 s | 1,774.28 s | 0.619 | 1,713.76 s |

- **The specdiff prediction holds**, and so does the prediction that the baseline is faster.
- **The [0.70, 0.96] band fails for imagick and lbm.** Both are faster than the STREAM ratio allowed. The band's lower
  bound assumed the 8 MiB streaming probe was the worst case, and it is not.
- **Clock check.** A wrong timebase on the new bitstream would scale every guest time. The host clock rules that out.
  The chain's host timestamps put the lbm stage at 1,438 s on the baseline, against 2,148 s (V1f stock) and 2,191 s
  (C.5a, P5). The difference, 710 s, matches the guest difference, 676 s, plus a faster boot. hart_cycles / 25 MHz
  agrees with the guest time on every stage.

**B2, read.** lbm dcnofprs takes 1,668.90 s against base 1,652.37 s, so it is 1.0 % slower. The base arms reproduced
their earlier records to 0.1 %, so this difference is outside the noise.

**Choice per benchmark** (rule: dcnofprs unless slower than base on the board):

| | Build | Capstone exe sha256[:16] |
|---|---|---|
| imagick | dcnofprs | f85865346235fd28 |
| nab | dcnofprs | b4eebed370ede16a |
| lbm | base | f0269eea1b555bd9 |

**Capstone against ordinary RISC-V, SPEC test, measured.** Both sides run with no heartbeat, each on its own SoC and
firmware.

| | Capstone, best build, Capstone SoC | Plain, baseline | **Capstone / ordinary RISC-V** | = SoC factor (Plain C.5a / Plain baseline) | × software factor (Capstone / Plain, same SoC) |
|---|---|---|---|---|---|
| imagick | 19.09 s | 11.17 s | **1.709** | 1.441 | 1.186 |
| nab | 871.88 s | 664.14 s | **1.313** | 1.169 | 1.123 |
| lbm | 1,652.37 s | 1,098.43 s | **1.504** | 1.560 | 0.964 |

- **The ≤1.20× target is not met on any of the three against ordinary RISC-V.**
- **The SoC factor dominates.** Plain, the same executable, runs 1.17-1.56× slower on the Capstone SoC than on the
  baseline. Both bitstreams start from the same upstream CVA6, so this is the cost of the fork's changes to ordinary
  code, before any Capstone instruction runs.
- **Mechanism: unresolved.** getppid (Addendum 44) shows the same instruction count at 1.86× the cycles, so the
  slowdown is per instruction, not extra work. Which fork change causes it is the next question, for RTL simulation:
  the same directed loops on both configurations.
- **Withdrawn.** Every "Capstone / Plain" ratio in this file before Addendum 44 measured the software cost on the
  Capstone SoC only, as Addendum 40 already said. None of them is a cost against ordinary RISC-V. The same-kit column
  is kept as the software factor.

## Addendum 46 (11 Nov 20:xx, before synthesis completes): where the SoC factor comes from. A V1f bitstream with the shadow-tag FSM tied off, Plain only

**Candidate, from the RTL.** An rtl-oracle read of the fork at 0ef470559 against the baseline named one change that
taxes ordinary code. I checked it in the source myself.
- It is the shadow capability-tag FSM in `core/cache_subsystem/wt_axi_adapter.sv`, and nothing guards it with
  CAPSTONE_EXT or CAPSTONE_ITFC_EN.
- `needs_tag` (line 156) holds for every D-cache load or store request in [0x8000_0000, 0xBC2D_2D2D), which is all of
  DRAM.
- A store takes a second 1-byte AXI write (TAG_WR).
- A load miss takes a second 1-byte AXI read (TAG_RD). Its data return is held until the tag beat arrives.
- Before that read, the load waits in TAG_WAIT for every outstanding tag-write response, with both AXI request lines
  held low.
- The arbiter grant is withheld in those states, so I-cache fills wait too.
- The D-cache is write-through, so every store goes through this path.
- The FSM is byte-identical in C.5a (1fc2cd5aa).

The oracle found no other plain-path difference that costs cycles:
- the cache geometry, cache type (WT), scoreboard, commit ports, predictors, MIG and SoC fabric match;
- the capability checks are gated on capmode.

It left store_unit, scoreboard, commit_stage, decoder and parts of csr_regfile UNRESOLVED.

**Measurement bitstream.** capstone-ariane branch measure-notag-v1f, 6806732fb, pushed: 0ef470559 (V1f) with
`needs_tag = 1'b0`, a one-term tie-off.
- Capability tags do not survive DRAM on it. It is for Plain timing only, never for Capstone code, and never an image
  for anything else.
- It is a measurement, not a Capstone configuration. It gives the upper bound of what a free tag mechanism would
  recover, for example a tag cache or skipping tag writes to granules already untagged.

**Run, after synthesis and seal.**
- Synthesis: launch-job.sh, full flow, 40 GB guard. It is held until the host's memory projection allows it.
- Kit: bkit41-stockfw-v1f with only BOARD_BITSTREAM changed (stock OpenSBI v1.3).
- Stages: tickcost, hiccup2, then Plain imagick, nab and lbm test.

**Predictions:**
- **(a)** If the tag FSM is the whole SoC factor, every reading lands within 5 % of the baseline: getppid 1,018
  cycles, imagick 11.17 s, nab 664.14 s, lbm 1,098.43 s.
- **(b)** If it is none of it, every reading lands within 5 % of V1f stock: 1,891 cycles, 16.89 s, 802.37 s,
  1,774.28 s.
- **What is reported** is the share of the gap it removes, per reading: (V1f − notag) / (V1f − baseline).
- **The lead question this answers.** It shows how much of Capstone's 1.31-1.71× against ordinary RISC-V is this
  implementation of tag storage, and how much is the rest of the fork.

## Addendum 47 (11 Nov 20:2x): gcc train01 Plain on the baseline read; overnight runs for mcf and xz, before they run

**gcc train01, Plain, baseline** (board 3, OpenSBI v1.3, no heartbeat): 1,736.35 s, specdiff PASS. The prediction
"below 3,280.89 s" holds.
- **Comparison, which needs a correction.** The Capstone record is 3,924.91 s: dcnofp 736f881f4958ac3b, kit bkit32-q32,
  run with the heartbeat. The Plain record on the Capstone SoC is 3,280.9 s, also with the heartbeat.
- **With the heartbeat removed by f** (Plain 0.0966, the mean f; Capstone f_C ≈ 0.088 from P2):
  - Capstone ≈ 3,580 s;
  - Plain on the Capstone SoC ≈ 2,964 s;
  - SoC factor ≈ 1.71;
  - **Capstone / ordinary RISC-V ≈ 2.06.**
- **This is an estimate.** A direct no-heartbeat Capstone gcc train run replaces it when one is made.

**The boards are released to this lane for the night** (the board lane holds no lease). Each driver passed a stubbed
dry run to its last stage.

| Board | Driver | Stages | Bitstream, kit |
|---|---|---|---|
| 1 | perf/ccsel/ccsel-mx.sh, mx1.list | Capstone mcf test: base 3633be7e2f887047, dcnofprs 741ba94c3e164e03 | C.5a, bkit23-all2-b1-nohb |
| 2 | perf/ccsel/ccsel-mx.sh, mx2.list | Capstone xz test: base 0d8878ad8a5046df, dcnofprs 66afc8c2d5227d33 | C.5a, bkit23-all2-b1-nohb |
| 3 | perf/stockfw/rvbase-p2.sh, rvbase-p2b.list | Plain xz test, Plain mcf test (hpm stage) | baseline, bkit42-rvbase-sbi13 |

- **ccsel-mx.sh** is ccsel-b2.sh with the list as an argument and one fix. ccsel-b2.sh did not export
  BOARD_URL_FILE, so its kit's board-run.sh would open board 1's console from any other board. The fix exports it
  per board. B2 ran on board 1 only, so none of its readings is affected.
- **Choice rule:** as in Addendum 45, dcnofprs unless it is slower than base on the board.
- **Predictions:**
  - Every stage passes specdiff.
  - Plain mcf and xz run faster on the baseline than on the Capstone SoC (Plain mcf test there, 9,077-9,150 s).
  - mcf is load-miss bound, and every miss pays the tag read (Addendum 46). So its SoC factor is at least 1.3, above
    nab's 1.17.
  - QEMU instret puts dcnofprs below base by 0.26 % for mcf. The board reading may fall either way, as it did for lbm.
- **Addendum 46's measurement** runs on the first board free after its bitstream is sealed. It is pre-registered
  there.

## Addendum 48 (11 Nov 20:2x): Capstone gcc train01, best build, no heartbeat, before it runs

This replaces Addendum 47's heartbeat-corrected gcc estimate with a measurement.
- **Stage.** perf/gcctr/stages/502.gcc_r-train01-capstone-lifetimes-c3fdcnofprs. It is a resolved copy of the
  3,924.91 s record's stage: no symlinks, no hardlinks, both checked. Only the executable is replaced, by c3fdcnofprs
  f95ad8ceddddc132 (perf/ccsel), and plan.json is updated to match.
- **Why c3fdcnofprs.** In the QEMU screen it has the fewest in-context instructions of the four gcc builds: −0.7 %
  against c3fdcnofp, −18.8 % against base.
- **Kit and board.** bkit32-q32-nohb, the record's kit without the heartbeat. C.5a, board 2, after Addendum 47's xz
  chain. Driver perf/gcctr/gcctr.sh is ccsel-mx.sh with the q32 kits and their module and launcher hashes. Its dry run
  is clean.
- **A record-keeping note.** The record stage's plan.json names 5642377943d724c6 (c3f). The record's image manifest
  shows that 736f881f4958ac3b (dcnofp) is what ran, so the record stands as published. Only the stage's plan.json is
  stale.
- **Predictions:**
  - specdiff PASS.
  - The time is 3,924.91 × (1 − f_C) ≈ 3,580 s, minus about 1 % for the exe change, so within ±5 % of 3,540 s.
  - Capstone / ordinary RISC-V = that time / 1,736.35, about 2.0-2.1.

## Addendum 49 (11 Nov 21:4x): the measurement bitstream is sealed; Addendum 46's run is launched

- **Seal.** caplifive_notag_6806732fb.bit, sha256 9c93016e5147de9edfa5c43fa3aab5363b68b1bc01055f76cc2e35e815c81bcd.
  - Built from capstone-ariane measure-notag-v1f 6806732fb = V1f 0ef470559 plus `needs_tag` tied to 0.
  - launch-job.sh full under vivado-guard: exit 0, 1 h 24 min, peak RSS 21.8 GB.
  - Placed 0444 in both bitstream stores; read back.
- **Checks against V1f (0ef470559):**

  | | notag | V1f |
  |---|---|---|
  | core-clock WNS @25 MHz | −12.827 ns | −13.873 ns |
  | routing errors | 0 | 0 |
  | LUTLP-1 in routed DRC | 0 | 0 |
  | core LUT | 154,176 | 154,688 |
  | wt_axi_adapter LUT / FF | 4,046 / 325 | 4,485 / 412 |

  The WNS difference is inside the ±3.74 ns null band. The adapter lost 439 LUTs and 87 FFs, so the tag FSM's logic
  is gone, as intended.
- **Run.** perf/stockfw/notag.sh on board 3, after the p2b chain (pid 86169), with kit bkit43-notag-sbi13 and
  notag.list. The stages and predictions are Addendum 46's, unchanged.
  - The dry run reached all 5 stages.
  - A wrong NB_SHA stops the driver at its hash gate (negative control).

## Addendum 50 (11 Nov 22:0x, before it runs): Plain perlbench and x264 test on the baseline; F11 from the RTL; the U1c8 candidate

**Plain perlbench and x264 test on the ordinary RISC-V baseline.** Neither has been run on caplifive_rvbase_d31345cf3.
- **Driver.** perf/stockfw/rvbase-p2.sh 3, waiting on the notag driver (pid 329593), with kit bkit42-rvbase-sbi13 (stock
  OpenSBI v1.3, no heartbeat) and list rvbase-p2e.list. The driver flashes the baseline back if notag is resident. Its
  dry run reached both stages.
- **Stages, executables, budgets.** Exe hashes come from the stage files, because plan.json is stale on both stages.

  | Stage | Exe hash | Step / booking |
  |---|---|---|
  | perf/plainreuse/stages/500.perlbench_r-test-plain-rv64r | 3bbd77760d40d34a | 55,000 s / 960 min |
  | perf/plainreuse/stages/525.x264_r-test-plain-rv64r | 2d8170fd12b5114f, imagevalidate 52b5c95397740354 | 60,000 s / 960 min |

  The kit's idle limit equals the step budget, so a quiet run is not cut.
- **Pass criteria:**
  - perlbench: rc 0, specdiff 2/2.
  - x264: rc 0. Its test stage has no imagevalidate, so specdiff reads 0/3 NO-OUTPUT by construction. The verdict is
    that the data outputs are identical by hash to the C.5a Plain record, out-b2q.
- **References** (C.5a, heartbeat):
  - perl: c0 499.35 s + c1 21,383.22 s (out-b1q). Its heartbeat share is f = 0.0946 (Addendum 34).
  - x264: gen 4,110.44 s + run 50,441.53 s (out-b2q).
- **Predictions.** The run command's time on the baseline is the C.5a time × (1 − f) divided by the SoC factor. That
  factor is 1.17-1.56 on imagick, nab and lbm.
  - perl c0 + c1: 12,700-16,900 s, point about 15,000 s.
  - x264 run command: 29,000-39,500 s, taking f ≈ 0.09 as for perl.
  - Outside either band means the SoC factor for these programs lies outside the range measured so far. That is a
    finding, not a fault.

**F11 from the RTL source.** Q-04 (b) is in the sealed sources of both bitstreams the Capstone arms use. In C.5a 1fc2cd5aa
and V1f 0ef470559, capstone_flu_unit.anvil:14-18 keeps an untagged (NOT_CAP) MOVC source, like NONLIN. Bitstream 3
(e7af48457) has the same lines, and its board reading was PRESENT. So `+movc-keeps-integer-source` matches the RTL on
C.5a and V1f by source. A board reading on C.5a is still to come. F11 stays open only for that reading.

**The U1c8 candidate.** sdk-U1c8 is the compiler lane's universal configuration:
- llvm-c4 2833be4b46d5 with U/FLAGS: shrinkto-globals with end-adjust, direct calls, no frame pointer, both
  drop-redundant-shrink modes, and -fno-jump-tables;
- libc and the rt-x runtime built with the same flags;
- the runtime's mapping cache bounded at 8 MiB;
- no Zbb. The baseline decodes Zbb (CVA6ConfigBExtEn = 1), but Zbb stays off in both arms.

None of it is landed. The c8 runtime line is uncommitted (wt-fpc a443914d9df2 + malloc-mapping.c:66). Before any U1c8 board
number is reported, the lane is asked to push it, and sdk-U1c8's directory hash goes in the record.

QEMU (kit-unq), in-context instructions:

| Benchmark | U1c8 | Best so far |
|---|---|---|
| perl test | 87.623 G, specdiff 2/2 (U/u2/twins/q-perl-test-U1c8/result.json) | udcnofp 91.12 G; the board's best perl, 28,509.41 s run command (out-udc-b3) |
| gcc test | 22.19 M | c3fdcnofprs 22.79 M (same kit, perf/u1c8) |
| gcc train01 | 12.452 G | c3fdcnofprs twin running |

Order, before any U1c8 run on a board:
1. **Same-kit QEMU screen.** dcnofprs twins of imagick, lbm, nab, mcf and xz (perf/u1c8/twin2.sh) sit beside the U1c8
   twins, all on kit-unq. U1c8 goes to the board for a benchmark only where it is not worse there.
2. **Shakedown boot on C.5a,** because rt-x has never run on silicon. Kit bkit23-all2-b1-nohb, the kit of the B2 base and
   dcnofprs runs. U1c8 gcc test comes first, then mcf test. It is pre-registered on its own, with its exe hashes.
3. **Then the choice runs:**
   - the short benchmarks first;
   - perl and x264 last, as single runs of U1c8 without the heartbeat. U1c8 is reported for perl or x264 if its run
     command is no longer than the best record × (1 − f). Otherwise the best record's build runs again without the
     heartbeat, and the shorter one is reported.

## Addendum 51 (11 Nov 22:3x, before it runs): U1c8 on the board, shakedown first, against dcnofprs

**The same-kit QEMU screen** (Addendum 50, step 1) is done. Kit kit-unq; capstone-lifetimes run commands only.
Instructions and entries are summed from result.json (perf/u1c8/capsum.py, which exits 1 on a missing reading). The
U1c8 lbm, imagick, nab and perl twins are the compiler lane's (c1-gate/gt2/U/u2/twins); the rest are perf/u1c8/twins.
Every run is rc 0, with specdiff PASS.

| Benchmark | U1c8 instret | dcnofprs instret | Δ | entries U1c8 / dcnofprs | Board |
|---|---|---|---|---|---|
| gcc test | 22,190,676 | 22,790,194 (c3fdcnofprs) | −2.63 % | 899 / 1,642 | shakedown |
| gcc train01 | 12,452,064,005 | 12,766,006,325 (c3fdcnofprs) | −2.46 % | 3,911 / 11,765 | train phase |
| nab | 7,165,821,787 | 7,181,950,384 | −0.22 % | 1,602 / 2,045 | yes |
| lbm | 6,097,844,994 | 6,092,805,233 | +0.08 % | 3,824 / 3,558 | yes (within noise, the board decides) |
| mcf | 26,784,991,655 | 26,968,269,803 | −0.68 % | 37,285 / 32,676 | yes |
| xz | 48,960,920,435 | 49,198,638,757 | −0.48 % | 47,795 / 38,461 | yes |
| imagick | 100,026,768 | 97,188,009 | +2.92 % | 376 / 727 | **no**: worse by more than 0.5 %, so it stays dcnofprs |
| perl | 87,623,200,705 | udcnofp 91,120,748,598 | −3.84 % | | Addendum 50, step 3 |

"Not worse" in Addendum 50 is read here as Δ ≤ +0.5 %. On mcf and xz U1c8 has more entries: its runtime's mapping cache
keeps memory mapped, which costs exits.

**Stages.** perf/u1c8/stages/<bench>-U1c8, built with perf/ccsel/mkstage.py from the same reference stages as the ccsel
candidates. They are one-arm and resolved. Only the lifetimes executable is replaced; helper programs stay at base.
- The executables are the compiler lane's stripped sdk-U1c8 builds, the same files the QEMU twins ran: gcc
  f2c29c060c8bfda4, nab 7828c50d35c5a29c, lbm 8b2d81e6899bbedb, mcf 442fc8659e51d917, xz c035ff64b4fc0f8e.
- Their sections match the board-run c3fdcnofprs gcc image: .capstone_virtual and .capstone_domreq are kept.

**SDK provenance:**
- **Runtime:** the c8 line is a443914d9df2 + 8f44f762e826 (origin/perf/heap-mapping-c8), MR_FREE_BYTES 64 → 8 MiB. That
  commit was read here.
- **Archives:** runtime 5aa897d3f1752657, libc lu1 a608a653ec7d66cc.
- **Compiler:** llvm-c4 from 2833be4b46d5 (origin/compiler-spec-c4).
- **The whole SDK:** perf/u1c8/sdk-U1c8.manifest lists capstone-cc, sdk.json and every file it names, each with its
  sha256. The manifest's own sha256 begins a59fede62c22b768.

**Runs.** perf/u1c8/u1run.sh is ccsel-mx.sh (Addendum 47) with only its log and output paths and this pre-registration
changed. It uses kit bkit23-all2-b1-nohb on C.5a, with no heartbeat. The dry runs reached every stage on both boards.
- **Board 2,** after gcctr (pid 128388). The list is u1b2.list:
  1. gcc test U1c8, the shakedown: the first silicon run of the rt-x runtime;
  2. nab U1c8, then nab dcnofprs;
  3. lbm U1c8, then lbm dcnofprs;
  4. xz U1c8. Its dcnofprs pair is Addendum 47's xz run on this board and kit tonight.
- **Board 1,** after the ccsel mcf chain (pid 86168): mcf U1c8. Its dcnofprs pair is Addendum 47's mcf run on that board.

**Predictions** (run command, board time):
- gcc test: rc 0, specdiff 1/1. A failure here is a silicon defect of the U1c8 runtime, not a timing, and it stops U1c8
  everywhere until it is understood.
- nab: U1c8 / dcnofprs in [0.97, 1.01]. lbm: [0.985, 1.015].
- mcf and xz: U1c8 / dcnofprs in [0.98, 1.01].
  - Their extra entries cost about 4,600 and 9,300 exits. At ~30 k cycles per exit that is 6 s and 11 s at 25 MHz,
    small against the run.
- specdiff PASS everywhere.

**Choice rule.** For each benchmark, the build with the shorter board time on its run command is reported, among base,
dcnofprs and U1c8. Every choice is disclosed with its exe hash. imagick stays dcnofprs, on Addendum 45's board reading.

## Addendum 52 (11 Nov 23:4x): Addendum 47's baseline readings; xz with the exit sweep off, before it runs

**Plain on the baseline** (board 3, perf/stockfw/rvbase-p2b.list, kit bkit42-rvbase-sbi13, no heartbeat; sums of the
run commands from result.json):

| Benchmark | Baseline | rc, specdiff | Plain on C.5a (record, with heartbeat) | SoC factor, raw / heartbeat-corrected (f 0.0966) |
|---|---|---|---|---|
| xz test, 12 commands | 4,956.00 s | 0 ×12, 12/12 PASS | 8,044.1 s | 1.62 / 1.47 |
| mcf test (hpm stage) | 5,085.28 s | 0, 2/2 PASS | 9,077-9,150 s | 1.78-1.80 / 1.61-1.63 |

- Addendum 47's predictions: both run faster on the baseline (holds); mcf's SoC factor is at least 1.3 (holds, 1.61
  corrected).
- **Capstone xz test, base 0d8878ad8a5046df** (board 2, C.5a, bkit23-all2-b1-nohb): 7,829.32 s, rc 0 ×12, specdiff
  12/12. **Capstone / ordinary RISC-V = 1.580.** dcnofprs is still running.
- **Where xz's time goes** (per command, run 0): 299 s in the context, 186 s outside it inside the VM window (3,458
  faults pinning 130,555 pages, 4,427 steps), and 64.9 s of exit teardown. In-context time over the 12 commands is
  4,578 s, below the baseline's 4,956 s total. The gap to the baseline is outside time.

**The run.** One xz test run on board 2 after its U1c8 chain (u1run.sh, pid 401469) ends, with `retire_exit_sweep=2`.
- **Build:** the fastest of base, dcnofprs and U1c8 on board 2's no-heartbeat runs. The rule is perf/ns/pickxz.py:
  done, 12 run commands at rc 0, 12/12 specdiff PASS, smallest sum of run seconds. Its controls: a candidate that is
  faster but fails one specdiff is skipped; no data exits 1 and the driver stops.
- **Stages:** perf/ns/stages/557.xz_r-{base,dcnofprs,U1c8}-ns, made by perf/ns/mkstages.py. Each is a resolved copy of
  its source stage. Every file is byte-identical except run.sh, which gains two lines and loses none:
  - after `@@SPEC-ARM-BEGIN`, write 2 to retire_exit_sweep through sysfs and print `@@NS` with the value read back;
  - before `@@SPEC-ARM-END`, write 3 back.
  - The executables are base 0d8878ad8a5046df, dcnofprs 66afc8c2d5227d33 and U1c8 c035ff64b4fc0f8e.
- **Kit and module:** bkit23-all2-b1-nohb, unchanged. Its module (2188cab26d40) declares the parameter 0644. Mode 2
  skips the sweep only on a destroying retire of an exiting task, and every other retire takes mode 0. xz has none of
  those: all 13 teardown lines of the base run read exiting=1. So one variable changes against the picked build's
  run.
- **Driver:** perf/ns/nsrun.sh waits for the pid, picks, writes perf/ns/ns-b2.list, and calls u1run.sh with log
  perf/ns/nsrun-b2.log and output out-ns2-b2. Its dry run is clean and is kept as perf/ns/dry-nsrun.log.

**Predictions.**
- rc 0 and specdiff 12/12. `@@NS` reads 2. The 12 command teardowns read `sweep=2 exiting=1`. Instret is within
  0.1 % of the picked build's run.
- Teardown falls from 778.5 s to about 279.5 s, the 5 Nov sweep-2 run's measure (out-ns-b2x: 1,566,694 pages,
  0.1784 ms/page, against 0.4968 ms/page at sweep 3).
- **The run sum is 400-600 s below the picked build's run** (point −499 s). For base that is about 7,330 s, or 1.48×
  the baseline.

**Disclosure, for the per-benchmark options table.**
- With mode 2, an exiting context's frames go back to Linux without the tag sweep.
- The tags are cleared by the kernel's overwrite before any reuse. results/2026-11-05-nosweep N1 showed this on
  silicon: 0 tagged granules in ≥ 47 k reused frames.
- Its gate weakness stands: the skip keys on PF_EXITING, not on `mm_users == 0`.
- The record for xz test already uses this mode (Addendum 7: bkit25-nset, 8,084.2 s with the heartbeat). This run is
  its no-heartbeat counterpart on today's builds.

**Not for mcf.** mcf retires its arrays during the run, and under mode 2 those retires take mode 0, which is slower
(results/2026-11-05-mcfx).

## Addendum 53 (12 Nov 00:3x): Addendum 46 read: the shadow-tag FSM is the whole SoC factor for Plain; instrumentation audit

**The run.** Board 3, bitstream caplifive_notag_6806732fb (V1f with `needs_tag = 1'b0`), kit bkit43-notag-sbi13, stock
OpenSBI v1.3, no heartbeat (HB lines 0 in every stage). It ran 23:22-00:21. Every stage rc 0, specdiff PASS.

| Reading | riscv baseline | notag | V1f stock | (V1f − notag) / (V1f − riscv) |
|---|---|---|---|---|
| getppid | 1,018 cycles | 1,054 | 1,891 | 95.9 % |
| imagick test | 11.17 s | 11.18 s | 16.89 s | 99.8 % |
| nab test | 664.14 s | 664.07 s | 802.37 s | 100.0 % |
| lbm test | 1,098.43 s | 1,098.38 s | 1,774.28 s | 100.0 % |
| hiccup2 gaps | 7.764 % | 7.756 % | | |

- **Prediction (a) holds on every reading:** each is within 5 % of the baseline.
- For ordinary code, the SoC factor is the shadow-tag FSM in `wt_axi_adapter.sv`, and only that.
- The optimisation that follows is docs/plans/2026-11-12-tag-page-bitmap.md, an on-chip page bitmap. RTL is on
  capstone-ariane perf/tag-page-bitmap from 1fc2cd5aa. It is lint-clean against its base, with every gated counter
  equal, and in simulation now.

**Capstone / ordinary RISC-V on tonight's clean runs** (C.5a, bkit23-all2-b1-nohb; sums of run commands from result.json):

| Benchmark | Capstone | riscv | Ratio | Capstone in-context | Largest bucket |
|---|---|---|---|---|---|
| xz (base) | 7,829.32 s | 4,956.00 s | 1.580 | 4,578 s | outside the context (pins, teardown) |
| mcf (base) | 10,555.03 s | 5,085.28 s | 2.076 | 9,306 s | in the context |
| lbm (base) | 1,652.37 s | 1,098.43 s | 1.504 | 1,383 s | in the context |
| nab (dcnofprs) | 871.88 s | 664.14 s | 1.313 | 816 s | in the context |
| imagick (base) | 19.16 s | 11.17 s | 1.715 | 11.8 s | outside the context and the launch |

**Instrumentation audit of the Capstone path.** The rule from here on:
- An instrument stays when it costs O(1) per command, or a few CSR reads per context entry.
- Anything that perturbs is off.
- No reported ratio uses a correction factor (f). Only direct runs count.

| Item | Cost inside the timed window | Status |
|---|---|---|
| batch heartbeat | f 8.9-10.8 % | OFF in every *-nohb kit; each run logs `HB lines 0` |
| launcher SHA-256 of the image at every exit (launcher a8a4dbe637ac758a under CAPSTONE_EXEC_DIAGNOSTICS=1, which boardvm.py exports) | gcc train 92 s, perlbench 7.0 ks | absent in the bkit23 (0bde5fd8e0c9c374) and bkit32 (1d8b4f0829bef0e7) launchers, where the switch gates only the fault record. A stage on an a8a4dbe6 kit must `unset` it |
| module bench counters and exit timing | a few CSR reads and atomic adds per entry; under 10^6 cycles per xz command | kept |
| per-command kernel lines (teardown, release, bench) and the launcher stats line | 3-4 lines per command | kept |
| monitor diagnostics (CPMX_DIAG, S23_REFUSAL_READ) | only on a refusal or fault; none of their tags in tonight's xz or mcf transcripts | kept |
| monitor per-invoke lines | about 5 ms each | OFF (CAPSTONE_SUPERVISE_QUIET) |

## Addendum 54 (12 Nov 01:4x, before it runs): the tag page bitmap bitstream on board 1; mcf U1c8 moves behind it

**What runs.** caplifive_tagbm_6df309173: capstone-ariane perf/tag-page-bitmap 6df309173, which is C.5a 1fc2cd5aa plus
the page bitmap of docs/plans/2026-11-12-tag-page-bitmap.md. It is being built now: launch-job full, 40 GB guard.

**Checks on the RTL before the board** (detail in the two commit messages):
- **Lint**, paired with 1fc2cd5aa: equal on every gated counter.
- **Directed test tag-page-bitmap: SUCCESS**, with and without S12_MEM_DELAY. Its positive control (the bitmap forced
  to ones) fails at T1, as designed.
- **claim-auditor.** It refuted one sub-claim of 52dab4ddd: tag_wr_pend could wrap after a fill. 6df309173 fixes that:
  - the fill drains to 0 before TAG_WR;
  - the counter is one bit wider.

  The simulation cannot create that wrap (T2b, recorded as non-discriminating), so the fix rests on the argument.
- **Synthesis-only gate on 52dab4ddd:**
  - the adapter grows from 1,657 to 1,980 LUT and from 412 to 530 FF, plus 7 RAMB36 and 1 RAMB18;
  - the core LUT count reads 145,134 against C.5a's 145,391.

**The seal is gated.** perf/tagbm/SEAL.sha256 is written only when all of these hold:
- the full build exits 0;
- routing is clean;
- WNS is within the null band of C.5a's;
- the 95-test sweep on 6df309173 shows no test whose classification differs from the 1fc2cd5aa sweep (same seed,
  S12_MEM_DELAY=12, SUP_TRACE; PASS/TIMEOUT/FAIL per test).

Until then the driver waits. A fault in any of these stops this addendum before any board call.

**Driver.** perf/tagbm/tbrun.sh on board 1, after the mcf base/dcnofprs chain (pid 86168). It waits for the seal, flashes
it, and reads it back. Then two chains in one bitstream session:
- **P:** Plain, stock OpenSBI v1.3, kit bkit44-tagbm-sbi13. These are Addendum 46's stages (tickcost, hiccup2, imagick,
  nab, lbm test), with no Capstone control.
- **C:** Capstone, kit bkit45-tagbm-all2-nohb, no heartbeat. These are the stages of the 11 Nov C.5a runs: imagick base,
  nab dcnofprs, lbm base, xz dcnofprs, mcf base. The kit's control rungs run in every boot, so they are this bitstream's
  Tier 1 for Capstone code.
- A failed P chain stops C.
- The kits differ from bkit42-rvbase-sbi13 and bkit23-all2-b1-nohb only in BOARD_BITSTREAM, which the driver checks.
- The dry run reached all 10 stages, and a foreign pre-registration commit stops it (perf/tagbm/dry-tbrun.log).

**The board move.** Addendum 51's mcf U1c8 run on board 1 (u1run.sh pid 401470, still waiting, no board call) is
stopped. It is queued again behind tbrun.sh with the same list, kit and output path.

**Predictions:**
- **P0.** Every stage rc 0 and specdiff PASS. The Capstone control rungs pass.
- **P1, Plain.** Every page is bit 0, because nothing stores a tag under stock OpenSBI. So each reading is within 3 % of
  the ordinary RISC-V baseline:
  - getppid 1,018 cycles (band 5 %);
  - imagick 11.17 s;
  - nab 664.14 s;
  - lbm 1,098.43 s.
- **P2, Capstone, against the 11 Nov C.5a runs.** Each run falls by at least 10 % on imagick (19.16 s), nab (871.88 s),
  lbm (1,652.37 s) and xz (7,740.77 s). Points: imagick ~14 s, nab ~730 s, lbm ~1,150 s, xz ~5,800 s.
- **P3, mcf.** It falls by less than 15 % (10,555.03 s, point ~10,000 s). Its node and arc arrays hold capabilities, so
  their pages stay on the tag path. That is the v2 question.
- **What would refute the design.**
  - P1 failing: the lookup costs more than the tag path saved.
  - P2 failing while P1 holds: Capstone's data pages are marked by tagged stores that this reading of the programs
    does not predict.

## Addendum 55 (12 Nov 02:5x, before any board call of Addendum 54): the seal checks read; one sweep flip, attributed

**Build.**
- 6df309173 built with launch-job full: exit 0, 1 h 14 min.
- Sealed bitstream: caplifive_tagbm_6df309173.bit, sha256 d85ef822a765d3511d310d67f32cd16ea4d079d3224ed9c10245ee282d73730c.
  The seal, with its reports, is in the build host's synth/tagbm-6df309173-full-out/seal. The bitstream is in the store at
  mode 0444, read back.

| | tagbm 6df309173 | C.5a 1fc2cd5aa |
|---|---|---|
| core-clock WNS @25 MHz | −13.908 ns | −13.758 ns |
| routing errors / LUTLP-1 | 0 / 0 | 0 / 0 |
| core LUT | 144,445 | 144,796 |
| wt_axi_adapter LUT / FF / RAMB36 + RAMB18 | 1,974 / 530 / 7 + 1 | 1,646 / 412 / 0 |

The WNS difference, 0.15 ns, lies inside the ±3.74 ns null band.

**The sweep.**
- Both sides: 95 tests, seed 20260922, S12_MEM_DELAY=12, SUP_TRACE, 2,000,000-cycle budget. Each run directory carries a
  PROVENANCE file. Compared with verif/sim/sweep-compare.py.
- 86 tests are identical in exceptions, readings and retired count.
- 3 tests are excused as missing on both sides (the random generators).
- 6 tests differ:
  - Five are TIMEOUT on both sides: call-hot, sbi-domain, domain-save-C, s06sec-ctx-scalar-roundtrip, r30-fill-init.
    Their counts differ, which a cycle-limited run does when its speed changes. Their classification does not.
  - **s07-ldc-chain-forward goes from TIMEOUT to SUCCESS** (1,400,919 cycles).
    - The base run's readings are a prefix of the tip's.
    - Both sides take the same single exception, the test's designed early one.
    - The base retires 120,120 instructions in 2,000,000 cycles; the tip retires all 131,333 in 1,400,919.
    - **Rerun of the base with a 4,000,000-cycle budget: SUCCESS at 2,185,643 cycles.** The flip is speed, not behaviour.
- **This misses Addendum 54's seal condition as written** ("no test whose classification differs").
  - That condition was meant to stop a functional regression. This flip is a passing test that now fits the budget.
  - The base passes it too, given time.
  - The seal is released on that ground, and this addendum is pushed before it.

## Addendum 56 (12 Nov 04:xx): Addendum 54's Plain chain read; P1 holds; the lookup costs one cycle per request

**Plain on caplifive_tagbm_6df309173** (board 1, kit bkit44-tagbm-sbi13, stock OpenSBI v1.3). Every stage ran with rc 0
and specdiff PASS, with no heartbeat lines.

| Plain test | riscv | notag | tagbm | tagbm / riscv | hart cycles, tagbm / notag |
|---|---|---|---|---|---|
| getppid (cycles) | 1,018 | 1,054 | 1,024 | +0.6 % | |
| imagick | 11.17 s | 11.18 s | 11.25 s | +0.7 % | 295.87 M / 293.75 M = +0.7 % |
| nab | 664.14 s | 664.07 s | 665.28 s | +0.2 % | 16.647 G / 16.616 G = +0.2 % |
| lbm | 1,098.43 s | 1,098.38 s | 1,112.84 s | +1.3 % | 27.835 G / 27.474 G = +1.3 % |

- **P1 holds**: every reading is inside its band (3 %; 5 % for getppid). Against V1f stock, the Plain SoC factor goes
  from 1.51 to 1.007 on imagick, 1.21 to 1.002 on nab, and 1.62 to 1.013 on lbm.
- The chain's two other readers are labelled "538.imagick_r ... rc!=0: 1, specdiff 0/1". These are stage-tickcost and
  stage-hiccup2, the tick probes; their reader uses imagick's name. The notag chain shows the same pattern
  (out-notag-n1-b3: 60.81 s and 68.65 s there, 60.85 s and 66.69 s here). They are not imagick runs and not failures.
- **The lbm residual has a mechanism in the RTL.** It is 361 M cycles on equal instret (6.925 G against 6.918 G).
  - `tbm_lkp_valid_d = ~dcache_data_empty & ~arb_ack[1] & ~tbm_clr_q` (wt_axi_adapter.sv:207) drops the lookup on
    every grant of the D$ port. So every D$ request in the data region waits one cycle, not only the first request to
    a page.
  - The plan doc said "one cycle per D$ memory request whose page is looked up for the first time". That is wrong, and
    it is corrected in this commit.
  - lbm, a store stream through a write-through D$, issues the most requests per instruction of the three.
  - This attribution is a reading of the RTL. The fix (below) is its test: if it is right, Plain lbm moves onto notag.
- **The fix, for the next bitstream:** keep the word index of the registered BRAM word, and treat the lookup as valid
  when the head's page lies in the same 64-page word and no bitmap write happened in the previous cycle. A head in the
  same 256 KiB as the previous one then pays nothing. The compare is 12 bits of the FIFO head's paddr against a
  register. needs_tag already feeds arb_gnt from the same paddr, so this adds no new cone.

**The board driver's wedge diagnostic now reads something the core does not.**
`run_sqlite_stages_fpga.py:3025-3034` and `:3247-3258` read DRAM tag bytes directly. On a page whose bitmap bit is 0,
those bytes are not the tags the core sees: the core reads every tag there as 0. A tag reported by that diagnostic on
this bitstream counts only for a page known to be marked. A search of the firmware and module sources finds no other
reader or writer of the tag region by address: caplifive-system and caplifive-buildroot, *.c/*.h/*.S, for
bc2d2d2d / 2d2d / tag_mem / 0xbff0 / 0xbc.

**Chain C** (Capstone, kit bkit45-tagbm-all2-nohb) started at 03:53 and is read against P0, P2 and P3 in the next
addendum.

## Addendum 57 (12 Nov 04:5x): Addendum 54's Capstone chain, first three stages read; P2 holds for imagick and lbm, fails for nab

**Capstone on caplifive_tagbm_6df309173** (board 1, kit bkit45-tagbm-all2-nohb, no heartbeat). Every stage has rc 0
and specdiff PASS. The kit's control rungs ran in each boot.

| Test (build) | C.5a, 11 Nov | tagbm | Change | P2 (≥ 10 %) | riscv | Capstone / riscv, C.5a → tagbm |
|---|---|---|---|---|---|---|
| imagick (base) | 19.16 s | 14.39 s | −24.9 % | holds | 11.17 s | 1.715 → 1.288 |
| nab (dcnofprs) | 871.88 s | 823.09 s | −5.6 % | **fails** | 664.14 s | 1.313 → 1.239 |
| lbm (base) | 1,652.37 s | 1,096.40 s | −33.6 % | holds | 1,098.43 s | 1.504 → 0.998 |

- In-context cycles: imagick 240.76 M, nab 19.552 G, lbm 23.800 G.
- **nab fails P2, in the way Addendum 54 named.** On Plain the bitmap removed 138 s of nab's tag path (V1f 802.37 s →
  664.07 s, notag). On Capstone it removed 49 s. The reading the addendum gave for this case: Capstone's nab data pages
  are marked by tagged stores. nab's molecule, strand, residue and atom records hold pointers, which are capabilities
  under Capstone, so their pages take the tag path. This is a hypothesis; nothing here has counted marked pages. It is
  the v2 question, as for mcf.
- **lbm now reads as fast as ordinary RISC-V, 0.998.** That is not a measurement artefact of this run. Addendum 40 already
  had Capstone lbm below Plain on the Capstone SoC, and that residual is still open. What is known:
  - both builds use the same clang (the SDKs' cc-prmerged) and the same OPTIMIZE; lbm base carries no Capstone-only
    tuning (RVC was ruled out as a cause in Addendum 36);
  - on the same tagbm bitstream, Capstone (1,096.40 s) is faster than Plain (1,112.84 s);
  - the candidates, not yet tested: address translation (Plain runs under Sv39 with 4 KiB pages and a 16-entry DTLB;
    a Capstone context does not translate) and the layout of lbm's two grids (musl mallocng against Capstone's heap).
  - If translation is the cause, a fairer ordinary-RISC-V arm is Plain with huge pages (THP), the same lever the audit
    listed for mcf. That arm is not run yet.
- **Fairness note, separate from these three.** U1c8 (Addendum 51) carries `-fno-jump-tables`, which is generic, not
  Capstone-specific. Plain has not been tried with it. Where U1c8 is reported (perl, gcc), Plain gets the same flag tried
  before the ratio is final. `-fomit-frame-pointer` in dcnofp is already clang's default for RISC-V at -O2, so Plain
  has it.

**Chain C continues:** xz dcnofprs, then mcf base. They are read against P2 (xz) and P3 (mcf) in the next addendum.

## Addendum 58 (12 Nov 08:0x): xz on the tag page bitmap (P2 holds); U1c8 on board 2 read and the choice rule applied; the tag-word cache (v2) through sweep and synthesis gate

**xz dcnofprs on caplifive_tagbm_6df309173** (board 1, Addendum 54's chain C, kit bkit45-tagbm-all2-nohb): 12/12
commands rc 0, specdiff 12/12 PASS. The run commands sum to 5,946.72 s, against 7,740.77 s on C.5a: −23.2 %. **P2 holds
for xz** (point ~5,800 s). Capstone / ordinary RISC-V for xz goes from 1.562 to **1.200** (riscv 4,956.00 s). mcf, the
last stage, is running; P3 is read in the next addendum.

**U1c8 on board 2** (Addendum 51; C.5a, kit bkit23-all2-b1-nohb, no heartbeat; every stage rc 0 and specdiff PASS):

| Test | dcnofprs | U1c8 | U1c8 / dcnofprs | Other record |
|---|---|---|---|---|
| gcc test | (not in this chain) | 13.2 s | | |
| lbm | 1,668.27 s | 1,673.20 s | +0.3 % | base 1,652.37 s (board 1, 11 Nov) |
| nab | 870.98 s | 876.85 s | +0.7 % | dcnofprs 871.88 s (board 1, 11 Nov) |
| xz (12 commands) | 7,740.77 s (board 1) | 7,797.03 s | +0.7 % | base 7,829.32 s |

- nab dcnofprs read 870.98 s here and 871.88 s on board 1: the two boards agree to 0.1 %.
- **Choice rule (Addendum 51: the shortest board time among base, dcnofprs and U1c8):** lbm base, nab dcnofprs, xz
  dcnofprs. U1c8 is not chosen for any of the three. gcc's choice waits for a gcc test dcnofprs run without the heartbeat.
- These are C.5a readings. The tagbm and tag-word-cache bitstreams change the memory side, not the code, so the order is
  expected to hold, but it is not re-measured there.

**The tag-word cache (v2) for marked pages.** Design: docs/plans/2026-11-12-tag-page-bitmap.md section v2, revised in
this commit after claim-auditor refuted the first draft. capstone-ariane perf/tag-page-bitmap:
- 7b929ac71: the page-bitmap lookup no longer waits a cycle for a head in the previous head's 64-page word (Addendum 56's
  lbm residual). Sim T5 matched pair, delay 0: 1,024 load misses 10,243 → 9,220 cycles, stores unchanged.
- dcb52a030: the TWC (256 whole tag words, BRAM). A marked-page load miss whose word is cached makes no tag read; a
  store whose cached byte is unchanged makes no tag write.
- ebe86cc01: claim-auditor on dcb52a030 found that an install could withdraw an AXI VALID without a handshake. Fixed,
  and a sim-only VALID-stability check was added. Its positive control (+define+AXI_DROP_PC) prints 9 drops.

Checks on ebe86cc01:
- **Directed test tag-word-cache:**
  - SUCCESS at S12_MEM_DELAY=12 and at delay 0;
  - the TWC_OFF control passes with the same values and 53 tag reads against the TWC's 25.
- **What no run has exercised.** The poison of a fill read by a tag or fill write issued while the read is in flight:
  16 race variants created no overlap, because the D$ issued the older store about 135 cycles after the younger load's
  tag beat. Nor did any test create the VALID-drop scenario. Both rest on review (claim-auditor, twice), not on a run.
- **Sweep, 95 tests, against the 6df309173 sweep** (same seed 20260922, S12_MEM_DELAY=12, SUP_TRACE):
  - 90 identical, 3 excused as missing on both sides.
  - call-hot and s06sec-ctx-scalar-roundtrip differ only in exception counts: TIMEOUT on both sides at 2,000,013.
  - RVFI: 91 of 92 traces identical; cap-tag-cache differs only in mcycle reads.
  - AXI-VALID-DROP 0 in every test, and a wait for READY was seen in each, so the check was not vacuous.
- **Lint gate PASS**, every counter equal to 6df309173.
- **Synthesis-only gate** (40 GB guard, peak 20.64 GB): wt_axi_adapter 2,706 LUT / 865 FF / 8 RAMB36 + 2 RAMB18,
  against 1,980 / 530 / 7 + 1 for v1's gate. Core 145,849 LUT against 145,134.
- The full build is running.

**Correction to the record of 6df309173.** The adapter's assertions, including tag_wr_pend_no_wrap, cited in Addendum 54,
sit inside `ifndef VERILATOR`. No simulation has ever evaluated them. 6df309173's wrap fix rests on the argument, as
Addendum 54 already said of the simulation; it does not rest on an assertion.

## Addendum 59 (12 Nov 08:1x, before it runs): the tag-word cache bitstream on board 1, after Addendum 54's chain; mcf U1c8 moves behind it again

**What runs.** caplifive_twc_ebe86cc01: capstone-ariane perf/tag-page-bitmap ebe86cc01, which is 6df309173 (Addendum
54's bitmap) plus the same-word lookup bypass (7b929ac71) and the tag-word cache (dcb52a030, ebe86cc01). Its checks are in
Addendum 58. The full build is running: launch-job full, 40 GB guard.

**The seal is gated.** perf/twc/SEAL.sha256 is written only when all of these hold:
- the full build exits 0;
- routing is clean, with no LUTLP;
- the core-clock WNS lies within the null band of 6df309173's −13.908 ns (±3.74 ns).

Until then the driver waits. A fault in any of these stops this addendum before any board call.

**Driver.** perf/twc/twrun.sh on board 1. It is tbrun.sh with only the bitstream, kits, lists and paths changed. It waits
for Addendum 54's driver (pid 788633, now in mcf) and for the seal, flashes the bitstream, and reads it back. Then:
- **P:** Plain, kit bkit46-twc-sbi13, perf/twc/tw-plain.list. These are Addendum 46's stages.
- **C:** Capstone, kit bkit47-twc-all2-nohb, no heartbeat, perf/twc/tw-cap.list: Addendum 54's chain C (imagick base,
  nab dcnofprs, lbm base, xz dcnofprs, mcf base), plus gcc test U1c8 and gcc test c3fdcnofprs after lbm.
- A failed P chain stops C.

The kits are bkit42-rvbase-sbi13 and bkit23-all2-b1-nohb with only BOARD_BITSTREAM changed, as the driver checks. The
dry run (perf/twc/dry-twrun.log) reached all 12 stages with the right kit for each.

**The board move.** Addendum 51's mcf U1c8 on board 1 (u1run.sh pid 788989, waiting, no board call made) is stopped. It
is queued again behind twrun.sh with the same list, kit and output path. U1c8 lost on lbm, nab and xz (Addendum 58), so
measuring the new hardware comes first.

**Predictions:**
- **P0.** Every stage rc 0 and specdiff PASS. The Capstone control rungs pass.
- **P1, Plain, against Addendum 56's tagbm readings.**
  - imagick 11.25 s and nab 665.28 s within 1 %; getppid within 5 % of riscv's 1,018 cycles.
  - **lbm moves onto notag: ≤ 1,105 s** (tagbm 1,112.84 s, notag 1,098.38 s). This is the test of Addendum 56's
    attribution of the lbm residual to the per-request lookup cycle. If lbm stays above 1,108 s, that attribution is
    wrong.
- **P2, nab dcnofprs: ≤ 782 s, at least 5 % under tagbm's 823.09 s.** This is the TWC's main test. A reading within 2 % of
  823.09 s means one of two things:
  - the marked-page cost is not mostly tag reads and writes of words the TWC can hold;
  - or nab's pages are not marked, and Addendum 57's hypothesis is wrong.
- **P3, imagick, lbm and xz: no worse than tagbm** (14.39 s, 1,096.40 s, 5,946.72 s; band +2 %). lbm and xz may fall by
  the lookup bypass.
- **P4, mcf: below its tagbm reading** (Addendum 54's chain C, still running), by at least 5 %. Its arrays hold
  capabilities, which is the TWC's case.
- **P5, gcc test U1c8: below its C.5a reading of 13.2 s.**

## Addendum 60 (12 Nov 09:4x): mcf on the bitmap (P3 holds); xz with exit sweep mode 2; the tag-word cache sealed; the flag-parity rule, jump tables, and two modes behind the name dcnofprs

**Names used from here on.** The old labels stay valid; they are the names in every addendum above.

| Name | Label(s) in this record | What it is |
|---|---|---|
| riscv-base | riscv, caplifive_rvbase_d31345cf3 | ordinary CVA6, Capstone off: the baseline |
| cap-hw0 | V1f (0ef470559), C.5a (caplifive_vp2c_1fc2cd5aa) | Capstone hardware before the tag work |
| cap-notag | notag (6806732fb) | measurement only: shadow-tag logic tied off |
| cap-pagemap | tagbm (caplifive_tagbm_6df309173) | + page bitmap |
| cap-tagcache | twc (caplifive_twc_ebe86cc01) | + same-word lookup bypass + tag-word cache |
| plain-on-cap | Plain on a Capstone bitstream | diagnostic only, never a baseline |
| cc-default | base | Capstone compiler, default options |
| cc-tuned | dcnofprs (c3fdcnofprs for gcc) | + direct calls, no frame pointer, drop-redundant-shrink |
| cc-universal | U1c8 | the compiler lane's all-in configuration |

**mcf base on cap-pagemap** (Addendum 54's chain C, last stage): 10,243.61 s, rc 0, specdiff 2/2 PASS, against 10,555.03 s
on cap-hw0, −3.0 %. **P3 holds** (a fall of less than 15 %). Capstone / riscv-base for mcf: 2.076 → 2.014 (riscv-base
5,085.28 s). Addendum 54 is complete: P0, P1, P3 hold; P2 holds for imagick, lbm and xz, fails for nab (Addendum 57).

**xz cc-tuned with retire_exit_sweep = 2** (Addendum 52's run; board 2, cap-hw0, kit bkit23-all2-b1-nohb): 12/12 rc 0
and specdiff PASS. The sum is 7,270.37 s, against 7,740.77 s with sweep mode 3: −470.4 s (−6.1 %; predicted −499 s). The
teardown reads 0.1783 ms per page (130,5xx pages, 23.28 s per command), on the model's 0.1784. Not yet run on
cap-pagemap.

**cap-tagcache sealed** under Addendum 59's gate:
- launch-job full exits 0 in 1 h 10 m, peak RSS 20.71 GB;
- route errors 0, LUTLP 0;
- core-clock WNS −12.315 ns against cap-pagemap's −13.908 ns, inside the band;
- core 145,134 LUT; wt_axi_adapter 2,695 LUT / 865 FF / 8 RAMB36 + 2 RAMB18;
- bitstream caplifive_twc_ebe86cc01.bit, sha256 52776f462203f7da1d33f55843d439552702ab5f29f5d3f8b8e614c1431ce1bf. It is in the
  store, read back. The seal holds the reports and a bundle of perf/tag-page-bitmap 1fc2cd5aa..ebe86cc01.

Addendum 59's driver began flashing it on board 1 at 09:35.

**Rule from the lead (12 Nov): every generic compiler option is the same in the Capstone builds and in riscv-base.**
Capstone-only options are exempt; ISA and ABI limits of the Capstone target are disclosed, not equalised. The audit:

| Option | riscv-base | Capstone | Status |
|---|---|---|---|
| -O2 | yes | yes | equal |
| jump tables | on (clang default) | **off**: the SDK wrapper passes `-fno-jump-tables` to every compile (capstone-cc:757) | unequal, against Capstone: fixed below |
| frame pointer | omitted (default at -O2 on RISC-V, clang CommonArgs.cpp:91-102) | omitted in cc-tuned, kept in cc-default | equal for cc-tuned |
| Zbb | off | off | equal |
| float ABI | lp64d | lp64 (+f +d), the Capstone target's ABI | disclosed |
| compressed instructions | rv64gc | none, the Capstone ISA; RVC ruled out as a cause (Addendum 36) | disclosed |
| compiler build | cc-prmerged clang | llvm-c1 / llvm-c4, same fork | the optimisation pipelines are yet to be compared |

- **Jump tables.** The wrapper's flag was a correctness workaround from August: a table read by a plain lw through a raw
  address. With `-fjump-tables` the current compiler reads the table through a capability derived from gp (scc,
  cincoffset, lw) and jumps with an integer jr.
- The compiler lane's census (c1-gate/gt2/U/RESULTS "JUMP-TABLE CENSUS") finds the same table dispatch already linked
  into every SPEC exe through musl and the runtime glue: printf_core, pop_arg, vfscanf, the delegate glue. Those exes
  passed on silicon. Silicon evidence that the integer jr form works there, though no trace shows those sites executing.
- **Correction to Addendum 57:** its note that U1c8's `-fno-jump-tables` should be tried on Plain is withdrawn. Every
  Capstone build carries the flag through the wrapper. The fix goes the other way: Capstone gets jump tables, as
  riscv-base has them.
- The fix is `-fjump-tables` appended to the build, after the wrapper's flag, so it wins. The shared SDK wrapper is not
  edited; the compiler lane is told before production flips it.
- Builds: perl, xz, imagick, nab and x264 change. lbm and mcf are byte-identical with the flag (no table-sized switch).

**Two modes behind "dcnofprs"** (found by the compiler lane, checked here).
- `-capstone-drop-redundant-shrink` was split into a definitions mode and a declarations mode on 8 Nov.
- gcc's c3fdcnofprs (f95ad8ce, built before the split) is declarations mode. Every perf/ccsel dcnofprs exe (11 Nov) passes
  the flag alone: definitions mode. So cc-tuned on the board so far meant declarations mode for gcc and definitions mode
  for nab, xz and mcf.
- Declarations-mode builds of the others (`-mllvm -capstone-drop-redundant-shrink-declarations` added; same compiler,
  98337e20741b, as every dcnofprs and jt build here):

| Test | SHRINK, defs → decl | Static instructions |
|---|---|---|
| perl | 28,190 → 18,374 | −5.2 % |
| nab | 3,599 → 3,385 | −0.7 % |
| xz | 1,751 → 1,651 | −0.5 % |
| x264 | 4,965 → 4,875 | −0.1 % |
| imagick | 32,351 → 32,290 | −0.02 % |
| mcf | 1,219 → 1,217 | −0.02 % |
| lbm | identical | identical |

- **cc-tuned means declarations mode from here on**, for every test. The board readings so far labelled dcnofprs for
  nab, xz and mcf are definitions-mode builds and are reported as such.
- QEMU twins are queued one at a time (perf/ccsel/qjt3.sh, kit-profq, under qemu.lock): declarations mode (D) and
  D + `-fjump-tables` (Djt) for imagick, nab, mcf, gcc, xz, x264 and perl.
  - The gcc pair was built by the compiler lane with c3f's recipe on today's compiler: D f07290e0aa4faa72 (it matches
    c3fdcnofprs instruction for instruction) and Djt bc46a5454508eb54.
- First QEMU reading, definitions mode: xz with `-fjump-tables`, 12/12 specdiff PASS, in-context instret
  49,589,636,311 against 49,198,631,731 (+0.79 %).
- The compiler lane recorded +0.44 % for gcc train earlier. Board time decides, not instret.

## Addendum 61 (12 Nov 09:4x, before it runs): board 2 flashes cap-tagcache and runs xz with exit sweep mode 2

**Why now.** Board 2 has been idle since 09:00. The next long runs, perl and x264 in declarations mode, wait for their
QEMU twins (perf/ccsel/qjt3.sh, now first in its queue), about two hours away. xz with retire_exit_sweep = 2 fills the
gap: its stage passed on board 2 (Addendum 60), and it measures the exit sweep on the final hardware.

**What runs.** perf/twc/tw2run.sh on board 2, which is twrun.sh (Addendum 59) with chain C only and list
perf/twc/tw2-cap.list:
- bitstream caplifive_twc_ebe86cc01 (sealed, Addendum 60), flashed and read back;
- kit bkit47-twc-all2-nohb, no heartbeat; its control rungs run in the boot;
- one stage: perf/ns/stages/557.xz_r-dcnofprs-ns, Addendum 52's stage. That is xz cc-tuned in definitions mode
  (66afc8c2d5227d33), with `echo 2 > /sys/module/capstone_vm/parameters/retire_exit_sweep` in the arm. Its dry run is
  perf/twc/dry-tw2run.log.

The xz build is definitions mode, as on board 1's chain. So against board 1's xz on cap-tagcache (Addendum 59, sweep mode
3) this pair differs in the sweep mode, and in the board.

**Predictions:**
- **P0.** rc 0, specdiff 12/12 PASS, control rungs pass; the transcript's teardown lines read sweep=2.
- **P1.** Teardown 0.17-0.19 ms per page, as on cap-hw0 (0.1783).
- **P2.** The 12-command sum is at most 5,550 s. That is cap-pagemap's 5,946.72 s minus Addendum 60's 470 s, with 2 %
  headroom. Capstone / riscv-base then reads at most 1.12.
- **P3.** Against board 1's xz on cap-tagcache (sweep mode 3, read later), this run is 400-550 s shorter. The two boards
  agreed to 0.1 % on nab (Addendum 58).

## Addendum 62 (12 Nov 11:0x, before it runs): the NI superpage fix for the Capstone line; a light heartbeat; mcf with 2 MiB THP on both arms (boards 2 and 3)

**1. An unequal fix, found and closed.**
- riscv-base (d31345cf3) was built with the five stock-CVA6 fixes the Capstone fork carries. One of them is missing from
  the cap-hw0 → cap-tagcache line: 213378a11, the NI probe's superpage PPN.
  - Without it, a load in the first 512 KiB of a 2 MiB page (and the low part of a 1 GiB page) is checked as an address
    below 0x8000_0000. It then takes the non-idempotent stall.
  - This affects kernel loads through the linear map, and any THP mapping, on Capstone only.
- The other four are ancestors of ebe86cc01: c77c65324, 6cbdaeeb4, cc1159a51, a73a7c670 + e7af48457, 2bf3d39f0.
- capstone-ariane perf/tag-page-bitmap:
  - d7ff26dc4: the niss-superpage test and the sim-only FPGA NI rule (from ca49a9c22);
  - 63e5dde82: the fix (from 213378a11).
- Matched pair, niss-superpage with CAPSTONE_FPGA_NI and SUP_TRACE:

  | | NI aborts | at ppn 0x40200 (2 MiB) | at ppn 0x6000 | bootrom control | cycles |
  |---|---|---|---|---|---|
  | before the fix | 330 | 132 | 66 | 132 | 10,935 |
  | after the fix | 132 | 0 | 0 | 132 | 10,198 |

- Lint gate PASS, every counter unchanged.
- Bitstream caplifive_twcni_63e5dde82 (cap-tagcache-ni): the full build is running. The seal file perf/thp2/SEAL-twcni.sha256
  is written only when all of these hold:
  - the build exits 0;
  - routing is clean, with no LUTLP;
  - WNS is within the band of cap-tagcache's −12.315 ns;
  - the running 95-test sweep shows no classification change against ebe86cc01's.

**2. A light heartbeat for every run from here on** (the lead, 12 Nov): perf/hb120.
- It is perf/nohb's tools with one change: the stage script prints `@@HB n`, a few characters, every 120 s, nothing else.
- The original heartbeat ran every 30 s, ran wc, tail and base64 on the step output, and sent about 330 characters.
- Seven beats fit in the watchdog's 900 s STALE window, so a silent stretch longer than that now means a hang.
- The same in every arm.
- The riscv runner wrapper is the hb120 copy of run-stage-noctl.py.
- Its cost is measured here (P5), not assumed.

**3. mcf with 2 MiB transparent huge pages, both arms.** This is the 6-7 Nov recipe (results/2026-11-06-thp,
2026-11-07-thp2) ported onto today's sources:
- the private THP-madvise kernel perf/thp/linux-board, relinked on each bake's rootfs. A relink test on today's rootfs
  passed: "AS usr/initramfs_data.o", Image built.
- **Capstone:**
  - capstone_vm 2188cab + b23b38c + 37806776 (wt-thp-all2 e73b31d80fc0), built against the THP tree: e9395bbf0f9570c9,
    no 2-byte instructions, `thp=1` in params;
  - capstone-vexec 9bf376a + 19bafe4, built `-DCAPSTONE_VEXEC_EXACT_IMAGE -DCAPSTONE_VEXEC_HUGE_HEAP`:
    f44ff6884dee5866. It madvises heap arenas of 4 MiB or more. The control built with EXACT_IMAGE only reproduces
    today's 0bde5fd8e0c9c374 byte for byte;
  - process-ABI module ctl-capstone.ko 5fe2b197 (the THP-tree build of f2a2645c).
- **riscv-base:**
  - the same kernel and process-ABI module;
  - capstone_vm m55-unify rebuilt against the THP tree (5020b6e9104eeb19). Its stock-tree rebuild reproduces
    63d60816's code exactly;
  - the stage sets `enabled=always` and `defrag=always`, because musl's malloc madvises nothing. Under madvise mode the
    riscv arm gets no huge pages at all; the 6 Nov Plain THP run was such a run, with no @@THP line.
- **Kits:**
  - bkit48-twcni-all2-nohb: bkit23 on cap-tagcache-ni;
  - bkit49-twcni-thp: bkit48 with THP;
  - bkit50-rvbase-thp: bkit42 with THP.
- **Stages** (perf/thp2/mkstages.py adds two lines to each and checks the diff):
  - Capstone mcf base reads back `@@THP enabled defrag thp` before the arm;
  - riscv mcf (the hpm stage of the 5,085.28 s reading) sets always/always and reads them back;
  - both print `@@THP-VMSTAT` (thp_fault_alloc and friends) after the arm.
- **Drivers:** perf/thp2/thrun.sh with b2.conf and b3.conf. Each runs the no-THP chain, then the THP chain, on one board
  and one bitstream. Dry runs reached all four chains (perf/thp2/dry-thrun-b{2,3}.log).
  - Board 2 waits for Addendum 61's driver and the cap-tagcache-ni seal.
  - Board 3 waits for Addendum 50's driver (x264).

**Fairness, stated before the run.**
- The two arms' huge-page policies differ:
  - Capstone madvises its large heap arenas;
  - riscv-base gets THP for every eligible anonymous mapping, which can only favour riscv-base.
- The launcher's heap is 2 MiB-aligned, and mallocng's mmaps are huge-page-backed only in their aligned interior, which
  favours Capstone. Neither size is measured; the vmstat lines show what each arm got.

**Predictions:**
- **P0.** Every stage rc 0 and specdiff PASS. Every THP arm prints `@@THP` with the mode set, and a thp_fault_alloc above 0.
  An arm with thp_fault_alloc 0 got no huge pages: it is void, not a negative result.
- **P1, Capstone mcf on cap-tagcache-ni without THP:** within 3 % of board 1's mcf on cap-tagcache (Addendum 59). The NI
  fix touches kernel loads, a small share of mcf.
- **P2, Capstone mcf with THP:**
  - at least 200 s under the same board's no-THP run, with its context entries falling by at least half (6 Nov: −94 %
    of fault exits, −343 s);
  - in-context cycles within 2 %. Huge pages change the faults, not the work.
- **P3, riscv mcf with THP:** between −5 % and +2 % against the same board's no-THP run.
- **P4.** Capstone / riscv-base for mcf is reported for the THP pair beside the no-THP pair, never instead of it.
- **P5, the heartbeat's cost:** riscv mcf without THP on hb120 is within ±0.5 % of Addendum 47's 5,085.28 s (no heartbeat,
  the same board, kit and stage).

## Addendum 63 (2026-11-12 17:00, before it runs): the whole SPEC test on cap-tagcache-ni with jump tables; mcf and the THP reading

**Readings since Addendum 62** (all rc 0, specdiff PASS unless stated):

| Run | Board, bitstream | Time | Capstone / riscv-base |
|---|---|---|---|
| gcc test U1c8 (cc-universal) | 1, cap-tagcache | 9.78 s | no riscv-base gcc test reading |
| gcc test c3fdcnofprs (cc-tuned) | 1, cap-tagcache | 11.67 s | no riscv-base gcc test reading |
| xz dcnofprs, sweep 3 | 1, cap-tagcache | 5,504.21 s (12 commands) | 1.111 |
| mcf base | 1, cap-tagcache | 8,552.7 s | 1.682 |
| mcf base, no THP | 2, cap-tagcache-ni | 8,551.98 s, entries 50,843 | 1.682 |
| mcf base, THP | 2, cap-tagcache-ni | 8,350.76 s, entries 25,789 | 1.642 (riscv THP arm pending) |
| x264 plain-rv64r | 3, riscv-base | gen 2,473.12 s + run 35,432.4 s | the denominator |

- Addendum 59: P3 holds for xz (−7.4 % against cap-pagemap); P4 holds (mcf −16.5 % against 10,243.61 s); P5 holds (9.78 s < 13.2 s).
- Addendum 61 P3 fails: sweep 2 is 221.6 s under board 1's sweep 3, not 400-550 s.
- Addendum 62 P1 holds (8,551.98 vs 8,552.7 s, 0.01 %): the NI fix does not move mcf.
- Addendum 62 P2: −201.2 s (holds, at the edge); entries −49.3 % (misses "at least half" by 0.7 points). In-context cycles
  and the @@THP-VMSTAT lines are read in the next addendum, with the riscv THP pair.
- x264 on riscv-base: specdiff 0/3 NO-OUTPUT by construction (no imagevalidate on the board stage, Addendum 50). Its data
  outputs equal out-b2q's by hash, 14 of 15; log.dec differs, as it does between any two runs (2026-10-28 Addendum 5).
  The run time is inside Addendum 50's 29,000-39,500 s band.
- Board 1's Addendum 51 chain (mcf U1c8 on cap-hw0, queued behind Addendum 59) started at 15:59. It is stopped: cap-hw0 is
  superseded and U1c8 was not chosen (Addendum 58). It is withdrawn, not read.

**Jump-table parity, checked by binary.** The lead's rule (Addendum 60) needs Capstone builds with `-fjump-tables`.
Rebuilt with perf/ccsel/build.sh, appending `-fjump-tables` after the wrapper's `-fno-jump-tables`:
- mcf base + jt = mcf base control, byte for byte (3633be7e2f88); mcf D = mcf Djt (1a26f98ed770). mcf has no jump table,
  so every mcf reading above already meets the rule.
- lbm base + jt = lbm base control (f0269eea1b55); lbm D = lbm Djt (6c973cd300e4). Same for lbm.
- imagick base + jt (f9e539160824) differs from base (f50dc465183e). imagick, nab, xz, gcc, perl and x264 need re-runs.
- QEMU twins (perf/ccsel/QLOG, qjt3): D against Djt, instret: perl −1.02 %, nab +0.16 %, imagick +0.005 %, gcc +0.27 %,
  xz +0.79 %; specdiff unchanged in every pair (perl 1/2 in both, the abigail command of Addendum 16).
  x264 does not finish in the 3,600 s QEMU step, as recorded on 11 Nov; both x264 twins are void.

**The run.** Bitstream cap-tagcache-ni (caplifive_twcni_63e5dde82, perf/thp2/SEAL-twcni.sha256), kit bkit48-twcni-all2-nohb,
the hb120 tools, no THP (the riscv-base references have none), exit sweep 3 except xz (sweep 2, the better mode,
Addendum 61). Driver perf/thp2/thrun.sh with perf/full/b{1,2}.conf.
- **Board 1** (perf/full/full-b1.list), after the Addendum 51 chain is stopped:
  1. gcc test c3fdcnofprsDjt (cc-tuned + jt);
  2. imagick base + jt (perf/full/stages/538.imagick_r-basejt) and imagick dcnofprsDjt; the faster one is imagick's choice;
  3. nab dcnofprsDjt;
  4. lbm base (no jump table, see above), so every test has a cap-tagcache-ni reading;
  5. xz dcnofprsDjt with sweep 2 (perf/full/stages/557.xz_r-dcnofprsDjt-ns, ns/mkstages.py's two lines, diff checked);
  6. perl dcnofprsDjt, last: its abigail command is expected to fail specdiff (Addendum 16).
- **Board 2** (perf/full/full-b2.list): x264 dcnofprsDjt, step 108,000 s.
- mcf is not re-run: board 2's 8,551.98 s is this bitstream, this kit and a binary equal to base + jt.

**Predictions** (against the cap-tagcache readings; jump tables are the only change besides the NI fix):
- **P0.** rc 0 and specdiff PASS everywhere except perl's abigail (1/2) and x264 (0/3 by construction; x264's data outputs
  must equal riscv-base's by hash, log.dec and .err aside). The xz arm prints `@@NS retire_exit_sweep=2`.
- **P1.** imagick, both arms, within ±3 % of 13.95 s.
- **P2.** nab within ±2 % of 755.05 s.
- **P3.** gcc test within ±5 % of 11.67 s.
- **P4.** lbm within ±1 % of 1,079.77 s.
- **P5.** xz within ±2 % of 5,282.61 s.
- **P6.** perl, both commands, 15,000-30,000 s (the board's best Capstone perl, udcnofp on cap-hw0, ran its long command
  in 28,509.41 s).
- **P7.** x264 run command 40,000-70,000 s.
- Outside a band is a finding, reported as read, not a fault.
- The riscv-base gcc test reading is still missing; a plain-rv64r gcc test stage is built and queued on board 3 in a later
  addendum.

## Addendum 64 (2026-11-12 17:20, before it runs): the riscv-base gcc test denominator; mcf diagnostics in QEMU

**riscv gcc test.** No riscv-base gcc test reading exists, so gcc's ratio has no denominator.
- Stage perf/full/stages/502.gcc_r-test-plain-rv64r: make-stage.py on spec/tree-plain (arm plain-rv64, rdctr
  6f05e7be00dc), with the executable swapped for the plain-rv64r build (cpugcc_r_base.plain-rv64r, 4d9ddde9aee9b35f,
  the one in the gcc train01 riscv stage) and plan.json updated to that hash.
- Board 3, kit bkit42-rvbase-sbi13, the hb120 riscv runner, after Addendum 62's driver (pid 1919310). Driver
  perf/thp2/thrun.sh with perf/full/b3.conf.
- **Prediction:** rc 0, specdiff 1/1 PASS, 5-10 s. Capstone's gcc test spends 4.70 s of its 11.67 s in context.

**mcf in QEMU (board-free).** The Capstone module's peak counts PINNED pages, and the adaptive fault-ahead pins up to
64 pages beyond a fault. Two QEMU runs of mcf base (perf/mcfdiag) read `fault_ahead_pages` and the peak with
fault_ahead_max 64 (control) and 0. If peak falls toward riscv-base's maxrss (306,160 kB, 74,540 pages), part of the
2.3x "footprint" is pinned-ahead pages, not data.

## Addendum 65 (2026-11-12 17:50, before it runs): what mcf's 2.3x memory is; plain-on-cap mcf on board 1

**Footprint, QEMU (perf/mcfdiag, mcf base, kit-profq), all rc 0 and specdiff 2/2 PASS.**

| Run | peak pinned pages | instret | entries |
|---|---|---|---|
| control, fault_ahead_max 64 | 174,308 | 27,039,956,421 | 32,510 |
| fault_ahead_max 0 | 174,231 | 27,044,640,437 | 392,533 |
| calloc without memset on fresh mmap groups (diagnostic SDK, wt-calloc) | 174,291 | 26,900,408,100 | 35,439 |

- The fault-ahead window does not inflate the peak: every pinned page is touched.
- The runtime's calloc (heap.c) memsets every block; musl's skips fresh mmap memory. Skipping it removes 0.52 % of
  mcf's instructions and leaves the peak unchanged. Not a footprint cause for mcf.
- **The realloc copies.** A host build of the same source with a malloc hook shows mcf growing arcs and sorted_arcs
  twice (to 2,102,404 and 4,102,404 arcs; riscv-base maxrss 306,160 kB). A diagnostic print in the runtime's realloc
  (QEMU) shows all four large growths MOVED (malloc + copy): 235,469,248 bytes twice, then 459,469,248 bytes twice.
  The Capstone mallocng path refuses an in-place remap when the group header size changes (realloc.c), so each growth
  copies; musl on riscv-base mremaps. arc_t is 112 bytes on Capstone against 72.
- So the 174k-page peak is the copy transient: the new arcs, the old and the new sorted_arcs alive at once (about
  3 x 235 MB). The steady footprint is about 1.56x riscv-base's in the arc arrays.
- Cost on silicon is not measured. A header-stable large-block layout (so the remap is taken) is a runtime change
  for a later addendum; it brings the allocator to what riscv-base's musl does.

**Plain-on-cap mcf.** riscv-base's mcf binary (the hpm stage of the 5,085.28 s reading) on the Capstone bitstream
splits mcf's in-context gap (about 2,650 s) into the hardware's share (the tag path on every load miss) and the
capability program's share (bigger structures, more misses).
- Board 1, after Addendum 63's driver (pid 2476894). Bitstream cap-tagcache (caplifive_twc_ebe86cc01,
  perf/twc/SEAL.sha256): Addendum 63 shows the NI fix leaves mcf at 0.01 %. Kit bkit46-twc-sbi13 (Addendum 59's
  plain-on-cap kit), the hb120 riscv runner, list perf/full/b1-plainmcf.list, conf perf/full/b1p.conf.
- **Prediction:** rc 0, specdiff 2/2. The reading R decides:
  - R within 3 % of 5,085.28 s: the hardware costs a plain miss-bound program nothing; mcf's gap is the capability
    program's;
  - R above 5,240 s: the tag path on misses is a share of the gap, (R − 5,085.28) / (8,552.7 − 5,085.28).
  - Plain-on-cap readings so far: lbm 1,105.04 s (riscv-base 1,098.43), nab 665.06 s (664.14).

## Addendum 66 (2026-11-12 18:30, before it runs): both arms on one shared NO_HZ_IDLE kernel (K2); Capstone quantum 32M; perl on the stock kernel withdrawn

**Why.** The lead's goal (12 Nov): every SPEC test benchmark under 1.20 (Capstone full temporal / riscv-base), with
equivalent conditions. Above it now: imagick (1.24), mcf (1.68); gcc, perl and x264 have no ratio yet. The largest prepared
lever is the tick replay (optimisation backlog C2): under HZ_PERIODIC the kernel replays every tick a context masked,
at each exit. K1 (quick Addendum, 8-9 Nov, C.5a): Capstone gcc train −3.6 %, Plain on the same kernel +2.0 %, ratio
1.212 → 1.145. The kernel is shared, so riscv-base must run on it too: both arms move to one kernel here, and every
riscv-base reference is re-measured on it.

**K2** (perf/k2): the board's 6.4.14 build-fpga tree, private copy, with K1's change (NO_HZ_IDLE, HIGH_RES_TIMERS
stays off) and the THP-madvise change of Addendum 62 (CONFIG diff perf/k2/config.diff). Image d0cf697ec434f328 on
perf/thp's seed, relinked per bake on the bake's rootfs. THP stays OFF in this addendum's stages (no MADV_HUGEPAGE
from either arm's launcher or libc, `enabled` left at madvise): K2 carries THP only so that a later THP pair needs no
new kernel.
- Modules on K2, each with a control: capstone_vm k2cap 009db2cafbf01beb (wt-thp-all2 e73b31d80fc0; .text identical to
  thpall2h e9395bbf, the THP-kernel build that ran Addendum 62); riscv-kit capstone_vm m55u-k2 c3ec55fbf4f59ec6 (.text
  identical to m55u-thp 5020b6e9); process-ABI ctl ko (fpga-pkg-127 source; .text identical to fpga-pkg-127's and to
  bkit49's 5fe2b197, 0 compressed instructions). The stock-tree control of capstone_vm (k2ctl) differs, as the THP
  config changes code, which is why the THP-kernel build is the reference.
- **Kits:** bkit51-k2cap = bkit48 (cap-tagcache-ni) with K2 and the firmware's CAPSTONE_SUPERVISOR_QUANTUM 8000000 →
  32000000 (backlog C3, Capstone's own scheduling parameter; same monitor d9da397); bkit52-k2rv = bkit42 (riscv-base,
  stock OpenSBI 1.3) with K2. Module params and launchers unchanged from bkit48 / bkit42.
- **First boots:** each kit's first stage is gcc test (seconds), so a broken kit costs one short boot.

**Lists** (perf/k2, driver perf/thp2/thrun.sh, hb120 tools on both arms):
- board 3, riscv-base, after Addendum 64's driver: gcc test, imagick, nab, lbm, xz, mcf (hpm stage), perl, x264;
- board 1, Capstone, after Addendum 65's driver: gcc c3fdcnofprsDjt, imagick dcnofprsDjt, nab dcnofprsDjt, lbm base,
  xz dcnofprsDjt with sweep 2, mcf base, perl dcnofprsDjt;
- board 2, Capstone x264 dcnofprsDjt, after Addendum 63's driver.
- Addendum 63's perl stage on the stock kernel (board 1) is withdrawn before it starts: its output directory is created
  empty, which chain-board skips. Its x264 (board 2, stock kernel) runs to the end.

**Predictions** (K1's shifts: Capstone outside time falls with the replayed ticks, riscv-base +0-2 %):
- **P0.** rc 0, specdiff as on the stock kernel, every stage on both arms.
- **P1, riscv-base on K2:** each benchmark within −1 % .. +3 % of its stock-kernel reference.
- **P2, Capstone on K2:** outside-context time (wall − in-context cycles / 25 MHz) falls on every benchmark; imagick
  13.87 s → 12.6-13.3 s; mcf 8,552 s → 7,900-8,300 s; in-context cycles within ±1 % of the stock-kernel run.
- **P3, the ratio on K2:** imagick ≤ 1.20; mcf 1.50-1.62 (still above 1.20: its in-context gap is footprint, Addenda 63/65).
- Each ratio is reported on K2 (both arms on K2) beside the stock-kernel ratio, never mixing kernels.

## Addendum 67 (2026-11-12 18:40): RETRACTION of xz sweep 2 as a configuration; the fairness audit; queue changes before they run

**RETRACTION.** xz with `retire_exit_sweep=2` (Addenda 52, 61, 63; reported as 1.066, "the better mode") is not a valid
Capstone configuration. The module's own comment (capstone_vm.c, retire_exit_sweep): "2: MEASUREMENT ONLY, no sweep at
all -- leaves tags in frames returned to Linux; never set outside a timing boot". A configuration that leaves tags in
freed frames is not Capstone. xz's ratio is the sweep-3 reading, 5,504.21 s / 4,956.00 s = **1.111**. Addendum 63's
xz stage (dcnofprsDjt with sweep 2, board 1, running) is measurement-only and not reported as a ratio. Every later xz
stage uses sweep 3: Addendum 66's Capstone list is changed to perf/ccsel/stages/557.xz_r-dcnofprsDjt (sweep 3)
before its chain started.

**The fairness audit** (claim-auditor, read-only, 12 Nov; load-bearing rows re-checked from their sources here). Rows
that are not inherent to Capstone:
1. **Mapping cache in riscv-base's libc only.** perf/plainreuse/musl-1.2.5/src/malloc/mallocng/plain_reuse.c (512
   mappings, 64 MiB), `T __plain_map`/`__plain_unmap` in musl-rv64gc-reuse/lib/libc.a, linked by plain-cc-reuse. The
   Capstone runtime of every measured build (c1-gate/sdk-dev-T3-mk, wt-perf 86f638759ecf) has no mapping reuse
   (`git grep MAPPING_REUSE` 0 hits; mallocng free.c munmaps). The cache was added as "the fair denominator" when the
   Capstone stack kept freed mappings; the measured Capstone builds do not. Favours riscv-base: on C.5a gcc −6.7 %,
   perl −6.0 %, imagick −2.2 % for Plain.
2. **Frame pointers and jump tables in the libraries.** Capstone's musl is built -O1 without -fomit-frame-pointer
   (survey-musl-capstone.py); its runtime has -fno-jump-tables in 82 files; riscv musl omits frame pointers. Favours
   riscv-base, unmeasured.
3. **Compiler identity.** riscv-base exes come from cc-prmerged (fca004e7), Capstone from llvm-c1 (724bf0be) and llvm-c4
   (98337e20), 300 commits apart with generic optimiser files changed. Direction unknown.
4. **PIE.** riscv-base compiles `-pic-is-pie` (checked with -###); Capstone static relocation. Favours Capstone.
5. **perl's harness on natperl** in the Capstone arm (one saved-context slot), on the perl under test in riscv-base.
   Favours Capstone, unmeasured.
6. **Best-of-N selection** for Capstone builds, one build for riscv-base. Favours Capstone, about the noise floor.
7. char signedness and predefined macros differ by target definition. Direction unknown, low.
Inherent to Capstone, disclosed: the launcher, arena setup and teardown inside the timed window; ticks masked inside a
context and the supervisor quantum; Capstone's bitstream fails timing at the shared 25 MHz (core WNS −13.426 ns) while
riscv-base meets it (+13.228 ns), so an iso-frequency comparison hides a frequency penalty; plain code on the Capstone
SoC +0.1-0.6 %. Checked equal: the SoC configuration (only CAPSTONE_* parameters differ), constraints, clocks (40 ns),
DDR, bootrom, kits' base manifest and dtb, SPEC inputs and argv, -O2 compile lines apart from the rows above.
Board-to-board: same-bitstream pairs agree within 0.10 %; no same-bitstream pair exists for the riscv-base bitstream.

**What follows.** One rebuild of both arms with the rows above made equal, then one campaign on K2 (Addendum 66's kernel):
- riscv-base: the same compiler binary as Capstone cc-tuned (llvm-c4), `-fno-pie`, musl without the mapping cache;
- Capstone: libc and runtime with -fomit-frame-pointer and jump tables; one compile configuration (cc-tuned + jump
  tables) for all eight, so no per-benchmark selection; sweep 3 everywhere;
- perl: both arms' harness on natperl, so the program under test is the only difference;
- QEMU twins of every executable before any board run; executable and library hashes in the next addendum.
Queue changes now, before they run: board 3's K2 chain keeps gcc test, imagick, nab and lbm (the kernel's effect on
riscv-base with today's executables; Addendum 66 P1); its xz, mcf, perl and x264 stages are withdrawn (empty output
directories, skipped). The two Capstone K2 drivers (boards 1 and 2) are stopped before any stage; they return with the
rebuilt executables in a later addendum.

**New readings** (rc 0, specdiff PASS): riscv gcc test 4.76 s (Addendum 64; prediction 5-10 s FAILS low); riscv mcf on
hb120 5,080.15 s (Addendum 62 P5 holds: −0.10 %); riscv mcf with THP 5,045.09 s (P3 holds: −0.69 %). gcc test is
therefore Capstone 11.79 s / 4.76 s = 2.48 on the stock kernel: in context 4.64 s, launcher 1.20 s, 1,035 exits about
3.85 s, outside the module about 2.1 s.

## Addendum 68 (2026-11-12 19:00, before it runs): the fair rebuild of both arms; gcc train replaces gcc test

**Revision of Addendum 67's plan, mapping cache.** Addendum 67 proposed removing the mapping cache from riscv-base. One
more fact: Capstone's gcc executables (c3f, c3fdcnofprs*) were linked against perf/arena/sdk-ac15, whose runtime has
CAPSTONE_HEAP_ARENA_CACHE=ON with the same bounds as riscv-base's plain_reuse.c (512 mappings, 64 MiB; heap.c AC_FREE,
AC_FREE_BYTES). So gcc had a cache on both arms and the other benchmarks on riscv-base only. The rebuild gives BOTH arms
the cache with equal bounds, which is the design plain_reuse.c states ("Plain baseline's counterpart"). ac15's other
switches: INDEX_BITS=16 and INDEX_SIMPLE (the slot index; no policy change) are kept; SLOT_CACHE (a LIFO reuse of freed
small slots, a policy musl's mallocng does not have) is left OFF.

**The lead's instruction (12 Nov): gcc is measured at train size (train01), not test.** gcc test's 4.8 s program is
dominated by fixed per-process costs (launcher 1.2 s, teardown 1.0 s); train01 runs about half an hour.

**riscv-base** (perf/fair): compiler llvm-c4 98337e20 (the Capstone arm's); `-fno-pie` (perf/fair/plain-cc-fairr);
musl 1.2.5 = perf/plainreuse's source with the cache (only free.c, glue.h, malloc.c, plain_reuse.c differ from T0b's),
rebuilt by llvm-c4 with the same configure line (-O1; CFLAGS_AUTO has -fomit-frame-pointer); runcpu label plain-rv64fr
(plain-rv64.cfg with only the label changed). Executables (sha256/16): perlbench eb432eec9edab726, gcc
600eed684dd7256b, mcf 05e07af3ffd0ac36, lbm a2501daf2852d28a, x264 c38b475c4339161b (imagevalidate
02f4f089cb1dad0c), imagick 73a8492dafeb50d9, nab 09db79dd881bd0b4, xz f0ba0c81d4970a20. Stages
perf/fair/stages/*-rv-fair2: the plain-rv64r stages with every binary swapped and plan.json updated; perl's command 1
runs its harness on natperl (`$^X="../sw/perl"`) exactly as the Capstone stage does.

**Capstone** (perf/fair/sdk2): wt-fair2 = wt-arena 46d67471e727 (86f638759ecf + 18 runtime commits, all default-off)
+ Application.cmake's three `-fno-jump-tables` turned into `-fjump-tables -fomit-frame-pointer`; libc with
MUSL_CAPSTONE_EXTRA_CFLAGS=-fomit-frame-pointer (frame-pointer set-ups in libc 1,199 → 95); all compiled by llvm-c4;
musl tree prepared by wt-fair2's prepare script (its 0009/0011 patches, private copy perf/fair/musl-arena);
HEAP_LIFETIMES=ON, SCRUB_FREE=OFF, ARENA_CACHE=ON, INDEX_BITS=16, INDEX_SIMPLE=ON; movc-keep as before (Q-04(b) is in
the bitstream line, F11). One compile configuration for all eight: cc-tuned (direct calls, -fomit-frame-pointer,
drop-redundant-shrink declarations mode) + -fjump-tables. Executables: perlbench 2fa975e3fece9054, gcc 7b431ca3e1d747db,
mcf 62e224bbe46fbc22, lbm edb0b4ef47145f36, x264 63076dc415c8bb52, imagick cb720928814d46d9, nab a266e1639caa7f1b,
xz ca23ca80633df2d1. Stages perf/fair/stages/*-cap-fair2 (perf/ccsel/mkstage.py; sweep 3 everywhere).
Compile-line check (nab): the two arms differ only by Capstone-only options, -fomit-frame-pointer and -fjump-tables,
both riscv-base's defaults at -O2.

**Runs**, all on K2 (Addenda 66/67), driver perf/thp2/thrun.sh, hb120 tools:
- board 3, riscv-base (bkit52-k2rv), after Addendum 67's short K2 list: gcc train01, imagick, nab, lbm, mcf, xz, perl, x264;
- board 1, Capstone (bkit51-k2cap, cap-tagcache-ni), after Addendum 65's plain-on-cap mcf: imagick, nab, lbm, gcc
  train01, mcf, xz, perl;
- board 2, Capstone x264, after Addendum 63's stock-kernel x264.
- Gate for the Capstone drivers: their QEMU twins (perf/fair/QLOG, kit-profq) rc 0 and specdiff as the previous builds
  (perl 1/2 is the known abigail command); x264 has no twin (3,600 s step). They are launched only after the twins read.

**Predictions:**
- **P0.** rc 0 and specdiff PASS on every riscv-base stage; Capstone as its twin.
- **P1, riscv-base fair vs the plain-rv64r references** (same programs, new compiler, no PIE, K2): each within −4 % .. +4 %.
- **P2, Capstone fair on K2 vs its stock-kernel reading:** imagick 12.6-13.5 s; nab, lbm within ±3 %; mcf 7,900-8,400 s.
- **P3, ratios:** lbm, nab, xz ≤ 1.20; imagick ≤ 1.20; gcc train01 ≤ 1.30; mcf 1.45-1.65 (above 1.20).

## Addendum 69 (2026-11-12 19:45, before it runs): the retraction extended to Capstone gcc train; shrinkto-globals for all eight; board 2's stock x264 stopped

**RETRACTION, extended.** The Capstone gcc *train* stages (perf/gcctr, perf/ga2 and their copies) write
`retire_exit_sweep=2` for the stage. Every Capstone gcc train reading taken with them (c3f 4,253.3 s and the rest) is
void, as Addendum 67's xz sweep-2 readings are. The test-size Capstone stages behind the reported ratios (imagick, nab,
lbm, xz dcnofprs, mcf, gcc test) write no module parameter (checked by grep of each run.sh). The fair gcc train01 stages
(perf/fair/stages/502.gcc_r-train01-cap-fair{2,3}) have that one write removed (sweep stays 3); they keep the stage's
other Capstone module settings, arena_pin=1 and step_budget_ns=10 s, disclosed as gcc-only Capstone tuning.

**gcc is reported at train01 only** (the lead, 12 Nov); gcc test leaves the table.

**Compile configuration.** The c3f recipe's Capstone-only pair `-mllvm -capstone-shrinkto-globals -mllvm
-capstone-shrinkto-end-adjust=1` is added for ALL eight (build-cap3.sh, executables *.fair3; .text −8 % gcc, −8.6 %
imagick, −4 % perl, −3 % x264), so the configuration stays one for all. QEMU twins: perf/fair/qtwins4.sh. Decision rule,
fixed now: fair3 is used if every fair3 twin passes as its fair2 twin; otherwise fair2 for all eight, and an x264 run
on fair3 is re-run on fair2.

**Board 2.** Addendum 63's stock-kernel x264 (old SDK and binary, started 16:37) is stopped: it cannot enter the fair
table and would hold the board until morning. Board 2 starts the fair x264 now (Addendum 68's list, stage
525.x264_r-cap-fair3, K2, bkit51-k2cap); its correctness check is the data outputs' hash against riscv-base's, as before.

## Addendum 70 (2026-11-12 21:15, before it runs): plain-on-cap mcf read; twins read; board 1's fair Capstone list, missing benchmarks first

**Plain-on-cap mcf** (Addendum 65; board 1, cap-tagcache, riscv-base's mcf binary): 5,180.26 s, specdiff 2/2. Against
riscv-base's 5,080.15 s (hb120): +1.97 %. Addendum 65's first branch (within 3 %) holds. The hardware's share of mcf's
gap is (5,180.26 − 5,085.28) / (8,552.7 − 5,085.28) = 2.7 %; the rest is the capability program (16-byte pointers:
arc 72 → 112 B; exits). A tag-path RTL change cannot bring mcf to 1.20.

**QEMU twins** (perf/fair/QLOG, kit-profq): fair2 all seven pass (perl 2/2, its abigail command now passes on this
runtime); fair3 (shrinkto-globals) imagick, nab, lbm, mcf, xz, gcc train01 pass; perl fair3 is running. A process slip:
two instances of qtwins4.sh ran (the first launch's pid was not the one killed); they split the work under the QEMU
lock; readings are functional only and unaffected.
**First fair riscv-base readings (K2):** gcc train01 1,731.62 s, imagick 11.09 s, nab 678.69 s (all specdiff PASS).

**Board 1** (the lead: fill the table first): perf/k2/k2f-cap.list, fair3 stages, order gcc train01, perl, mcf, imagick,
nab, lbm, xz; kit bkit51-k2cap (cap-tagcache-ni, flashed by the driver), K2. If perl's fair3 twin does not pass, its
stage is withdrawn before it starts (Addendum 69's rule then moves all eight to fair2).

## Addendum 71 (2026-11-12 22:45, before it runs): Capstone gcc train fails on the 65,536-record lifetime table; an 18-bit kit

**Reading.** Board 1, gcc train01 fair3 on bkit51-k2cap: rc 1, specdiff 0/1, after 2,130.48 s (instret 8.63 G of the
twin's 13.25 G). CAPSTONE_VM_STATS: node_capacity 65,536 (node_bytes 1 MiB). Its QEMU twin passed on kit-profq,
node_capacity 262,144, and the last passing board gcc train (out-gcctr-b2, kit bkit32-q32-nohb on vp2c_1fc2cd5aa) read
node_capacity 262,144. The K2 module k2cap was built without CAPSTONE_VM_INDEX_BITS, so 16 bits; gcc runs out of
lifetime records, malloc fails, gcc exits 1 (gcc's stderr was not in band; the mechanism is inferred from the two
capacities and the exit code, not read). A kit error of mine: the module's index width was not checked against the
workload's need.

**Fix.** capstone_vm k2cap18 0cf6bc2c892fc993: the same source and K2 tree with -DCAPSTONE_VM_INDEX_BITS=18 (option
A; the module's comment: the width must be the bitstream's; C.5a-line bitstreams carry it, bkit32 ran 18 on
vp2c_1fc2cd5aa and cap-tagcache-ni descends from that line). Kit bkit53-k2cap18 = bkit51-k2cap with that module.
The table width is a capacity, not a speed setting.

**Queues.**
- Board 1: perl fair3 (running on bkit51; perl's twin uses about 16 k nodes) finishes; the rest of its list is
  withdrawn before starting. Then perf/k2/k2f18-cap.list on bkit53: imagick (the new module's first boot), gcc train01,
  mcf, nab, lbm, xz.
- Board 2: the fair x264 on bkit51 (started 19:38) is stopped: its node need is unknown, a failure would show only at
  the end, and its module would differ from the other benchmarks'. Restarted on bkit53 (imagick first, then x264).
- perl alone runs on the 16-bit table; it is disclosed, and re-run on bkit53 if the boards have room.

## Addendum 72 (2026-11-13 03:10, before it runs): the Capstone arm is option B + native; names; one kernel K2N; riscv-base on it

**The lead's decision (13 Nov), recorded:** the Capstone arm is **option B** (55-bit lifetime ids) through the **native
route**, as Addendum 10 chose and as the paper describes. On 12 Nov the work ran on the 30-bit-id line through the
module route by my mistake. Every Capstone reading of Addenda 54-71 is relabelled **cap-ids30(-tagcache), module route,
diagnostic**; none enters the final table.

**Names** (the lead: name a bitstream by the ideas it combines):
riscv-base (d31345cf3); cap-ids30 (1fc2cd5aa, formerly "C.5a"); cap-ids30-tagcache (63e5dde82); cap-ids55-native
(0ef470559, formerly "V1f": 55-bit ids + native Linux-task support + R-70 fix, NI fix included);
cap-ids55-native-tagcache (79cf8de17 = cap-ids55-native + page bitmap + tag-word cache; capstone-ariane
perf/v1f-tagcache; the adapter file byte-identical to cap-ids30-tagcache's; lint gate PASS, figures as 63e5dde82's);
cap-ids55-native-pagemap (c62d59175, page bitmap only).

**Bitstream order, fixed now:** cap-ids55-native-tagcache (building on apollo since 02:30) if it routes, seals and passes
the native smoke as its first boot; else cap-ids55-native-pagemap (building on the synth lane's host); else the sealed
cap-ids55-native (caplifive_e5nids55f_0ef470559). Each native chain starts with the native smoke stage.

**One kernel for both arms: K2N** (perf/k2n/mk-k2n.sh): the native ids55 board kernel tree (native01/linux-board-ids55:
6.4.14 + the linux-6.4/ native series, RISCV_CAPSTONE_TASK, capstone_vm built in, the native command line incl.
cma=256M and node_flat_records=262144) with K2's tick change (NO_HZ_IDLE, HIGH_RES_TIMERS off), no THP. Config diff
against the native tree: NO_HZ only (perf/k2n/config.diff). Image fd61316e8466983367.
- riscv-base on K2N: kit bkit54-k2n-rv (bkit52-k2rv with K2N relinked per bake, ctl ko built on K2N with .text equal to
  fpga-pkg-127's, no module parameters). Its tools perf/k2n/tree are perf/nohb/tree with one change: boardvm's insmod
  guard tests /sys/module/capstone_vm (a built-in module is not in /proc/modules). No heartbeat, as the native kits
  run none (Addendum 62's hb120 cost on riscv mcf was −0.1 %). Real bake and relink dry run: rc 0, gate PASS.
- Disclosed: K2N's native command line and cma=256M apply to riscv-base too.

**Mapping cache: off on both arms.** The native runtime (dev's allocator) has no mapping cache; riscv-base therefore
runs plain-rv64f (llvm-c4, -fno-pie, musl WITHOUT plain_reuse.c), stages perf/fair/stages/*-rv-fair. Cache on both
arms would need a port of the dev-allocator mapping reuse (perf/heap-mapping-c8) into the native runtime; that is an
improvement step, not done here.

**Board 3, stopped and restarted:** its K2 riscv list (Addendum 68) is stopped during perl (the K2 kernel is no longer
the measurement kernel). It now runs perf/k2n/k2n-rv.list: imagick, nab, lbm, gcc train01, mcf, xz, perl, x264.
Boards 1 and 2 keep their cap-ids30 module-route diagnostics until the native stages are ready, then switch.

**Predictions, riscv-base K2N no-cache vs its K2 cache reading (Addendum 68):** imagick +0 % .. +4 % (the cache was
worth −2.2 % on cap-ids30 Plain); nab, lbm, mcf, xz within ±1.5 %; gcc train01 +3 % .. +9 % (cache −6.7 %).
Outside those bands the kernel differs from K2 for riscv-base, which is a finding.

**Native route caveat, to be carried in every native row:** checked accesses are emulated by the firmware (native
CSCHECK emulation, about 600 cycles each, MEASUREMENT ONLY shortcuts on frame-slot and epoch trust, not skipped work).

## Addendum 73 (2026-11-13 04:05, before it runs): the Capstone arm on cap-ids55-native-tagcache, native route, K2N

**Bitstream sealed** (board lane): caplifive_ids55-native-tagcache_79cf8de17.bit, sha256 f56a175c44983eea…, 11,443,722 B
(capstone-ariane 79cf8de17 = cap-ids55-native + page bitmap + tag-word cache). Routed clean: WNS −14.789 ns, TNS
−276,131, WHS +0.055, Slice LUTs 179,560 (88.11 %), placer congestion 4, router 5 (Route 35-447 printed: closer to the
edge than cap-ids55-native, which routed at 4). LUTLP-1 0, loop headers 0. Lint gate PASS (as Addendum 72). The
synth lane's hedge (page bitmap only) was stopped once this one routed.

**Native SDK and images** (compiler lane): origin/nat-fair-sdk 8decdcbc6aaf on native/c5a-ids55 441d7ec409ce: llvm-c4,
libc and runtime -fomit-frame-pointer (jump tables were already on in this tree), sdk-native-c1's cmake options, no
mapping cache, movc-keep ON. Images (perf/nat-fair/exes, .strip.dom sha256/16): perlbench b3ad5fa91d515349, gcc
a73396ab7b6f9ff6, mcf 5e9d25039d49b7bc, lbm c3eac87b6cc352e5, x264 66b242b48f5e6159, imagick 22a6ca7711a8460d, nab
e2d3a3189fd9cd8a, xz 071b49f35c13fe3e; compiled with cc-tuned + jump tables + shrinkto-globals. QEMU twins (qemu55,
native route): lbm, mcf, nab, xz, gcc test rc 0 and specdiff PASS; imagick's run passes and its native imagevalidate
fails (rc 255; the board stages below run imagevalidate as riscv-base's plain build); perl 1/2 on the twin's own stage
(harness from a different stage); x264 and gcc train01 running.

**Board stages** (perf/natstage): perf/fair/stages/*-cap-fair3 (SPEC's commands and inputs) converted by
make-native-stage.py (native/c5a-ids55), with imagevalidate as the plain riscv build and perl's harness on natperl with
$^X = the native perl image; then the kit's make-train-stage.sh (native smoke first). The timed window brackets only
the program's exec. Largest image (gcc train01): firmware 23,760,392 B, under the 26,000,000 B cap (real bake).

**Kits** (board lane): bkit56-k2n-native-tagcache (this bitstream) and bkit55-k2n-native (cap-ids55-native, the
fallback), both on K2N via a private copy of the tree (same .config and command line as riscv-base's K2N:
node_flat_records 262144), monitor csbi-ids55 4f51541, quantum 8M, deep dry runs PASS.

**Runs** (driver perf/natstage/nat-run.sh: flash and readback, per stage a booked boot; the smoke must PASS):
- board 1: imagick (the bitstream's first boot), gcc train01, mcf, perlbench;
- board 2: nab, lbm, xz, x264. Board 2's cap-ids30 module-route x264 diagnostic is stopped for it.
- riscv-base on K2N so far (no cache, specdiff PASS): imagick 11.26 s, nab 675.63 s, lbm 1,077.88 s (Addendum 72's
  bands hold: +1.5 %, −0.45 %, −0.3 % against K2).

**Predictions** (Capstone native, cap-ids55-native-tagcache, K2N, against riscv-base on K2N): imagick ≤ 1.15 (native
skipped the launcher costs on cap-ids55-native: 0.89 of module in-VM); nab, lbm ≤ 1.10; xz ≤ 1.15; gcc train01 ≤ 1.35;
mcf 1.4-1.6 (its gap is footprint, Addenda 63/65); perlbench and x264 no band (no native test reading exists). Every
native row carries: checked accesses emulated by the firmware.

## Addendum 74 (2026-11-13 06:15, before it runs): first native readings; perl test on makerand; the mapping cache on both arms for gcc, perl, imagick

**Readings, cap-ids55-native-tagcache, native route, K2N, no mapping cache on either arm** (specdiff PASS everywhere;
riscv-base on K2N, plain-rv64f): imagick 13.75 / 11.26 s = 1.221; nab 747.14 / 675.63 = 1.106; lbm 1,073.62 /
1,077.88 = 0.996; gcc train01 3,409.84 / 1,879.48 = 1.814. riscv-base mcf 5,146.72 s. Addendum 73's predictions:
imagick (≤ 1.15) and gcc (≤ 1.35) FAIL; nab and lbm hold. Native counters on gcc train01: 46,452 service escapes
(mmap/munmap; no mapping cache in the native runtime) and 32,033 fault escapes, against 104 and 239 on imagick.
Simulation sweep of the port (99 tests, cap-ids55-native vs + tag adapter, one file different): RVFI 93/96 identical;
the 3 that differ do so by construction (two page-bitmap tests read a forged tag that the bitmap hides, and
cap-tag-cache reads mcycle).

**perl test on the native route is compared on makerand.pl only, on both arms.** test.pl runs its sub-tests in t/, and
a native task's grant is rooted at its start directory, so ../TestInit.pm is refused ("Permission denied"; compiler
lane's twin, 13 Nov). The native stage's test.pl command is removed before the board ran it; riscv-base's perl row
uses its makerand command (run0) from the same stage it always ran.

**Mapping cache on both arms (the improvement step for gcc and perl):** native SDK nat-fair-cache (compiler lane
eb73a2bf475c = nat-fair-sdk + origin/perf/heap-mapping-reuse-review 61fa2db9714c, MAPPING_REUSE on, REGION off,
MR_FREE 512 whole mappings, 64 MiB, exact-length reuse, zeroed; also revokes each released mapping — Capstone's own
lifetime semantics); riscv-base plain-rv64fr (plain_reuse.c, 512 / 64 MiB). QEMU twins: gcc train01 service escapes
46,452 → 2,600, fault escapes 24,317 → 2,594, specdiff PASS; mcf, xz, perl makerand PASS.
- Board 1, after its running chain: Capstone with the cache (imagick, gcc train01, perl makerand), then reflashed to
  riscv-base: riscv-base with the cache (imagick, gcc train01, perl), both on K2N.
- **Predictions:** Capstone gcc train01 with the cache 2,000-2,800 s; riscv-base gcc train01 with the cache
  1,700-1,780 s (its K2 reading 1,731.62 s); gcc ratio 1.15-1.60. imagick and perl within ±3 % of their no-cache
  ratios.

## Addendum 75 (2026-11-13 07:30, before it runs): the native I/O alignment batch (bkit57); board 1's cache chain moves onto it

**Reading since Addendum 74:** Capstone xz test (native, no cache) 5,504.98 s over its 12 commands, specdiff 12/12
PASS (riscv-base's K2N xz still running on board 3).

**gcc train01 instruction profiles on the QEMU twin** (capmix3, the train01 command's window, specdiff PASS each;
perf/gccprof): native, no cache: user 12.42 G, kernel 1.83 G, firmware 0.18 G = 14.43 G; native, cache: user 12.59 G
(+165 M of memset: reused memory is zeroed in user space), kernel 0.43 G, firmware 0.04 G = 13.06 G; riscv-base
(rv-fair, kit-profq kernel, not K2N): user 10.78 G, kernel 0.82 G, firmware 0.08 G = 11.68 G. So instructions give
1.235 (no cache) and 1.118 (cache), against the board's 1.814: most of gcc's gap on silicon is not instruction count.
The user +15 % is capability arithmetic and bounds operations (+1.78 G; pointer loads/stores become capability
ones about one for one), per the compiler lane's per-function diff (perf/nat-fair/gcc-func-diff-full.txt).

**bkit57-k2n-native-tagcache-ioalign** = bkit56 with the native kernel copy perf/k2n/linux-board-k2n-native-ioalign:
linux-board-k2n-native plus the five adapter files of llvm-capstone native-io-align 2d380284ee86 (base native/c5a-ids55
441d7ec409ce; the copy's five files equalled the base's before). Changes, all in Capstone's own native path
(capstone_vm; riscv-base's K2N tree is not touched):
- 517f8ec5044a: kernel read/write buffers placed at the file position's alignment. imagick's 864 KB output went
  through a byte-by-byte memcpy because the TGA header leaves the file offset misaligned; twin: that loop 858,448 ->
  10,180 iterations, kernel instructions of the convert command -17.0 %, program instructions unchanged.
- 1931fe59535e: cu_copy skips the zero-length probe check once the next cursor is predicted (the authorizing check
  still covers exactly the chunk; on a mismatch the old two-check path runs). Twin imagick checks 3,169 -> 2,977.
- 2d380284ee86, MEASUREMENT ONLY: native counters gain checks, check_cycles, runs, run_cycles, between_cycles (cycle
  CSR, read with interrupts off; the firmware sets mcounteren to all ones).
Checks: twins on the batch kernel give byte-identical result files for imagick, xz, gcc test, gcc train01 (compiler
lane); the copy compiles drivers/misc/capstone with the same 11 pre-existing warnings; native-tree.sha256 regenerated
over the same 67 paths, exactly the five files differ; stubbed dry runs of both chains reach their last stage.

**Runs, board 1** (after its running chain: mcf, perl makerand). Addendum 74's Capstone cache chain on bkit56 was
stopped before it booked anything and replaced by:
- d: bkit57, no cache: imagick (the isolated effect of the batch against Addendum 74's 13.75 s);
- e: bkit57, mapping cache (nat-fair-cache): imagick, gcc train01, perl makerand;
- then riscv-base with the cache (Addendum 74's chain, unchanged): imagick, gcc train01, perl.

**Predictions:** d imagick 13.15-13.55 s (ratio 1.17-1.20 against 11.26 s); checks delta per command = its
emu_fingerprints delta (every check emulated), about 2,970-3,000 on imagick; check_cycles / checks 1,000-20,000 cycles
(informational: the first silicon reading of an emulated check). e gcc train01 2,000-2,800 s (Addendum 74's band), with
service + fault escapes under 8,000; imagick and perl within ±3 % of d's and of the no-cache perl ratio.

## Addendum 76 (2026-11-13 07:45, before it runs): cap-ids55-native-tagcache-mlp against cap-ids55-native-tagcache, board 2

**Reading since Addendum 75:** riscv-base xz on K2N (no cache) 4,993.81 s over 12 commands, specdiff 12/12 PASS; the
Capstone native xz row is 5,504.98 / 4,993.81 = 1.102 (Addendum 73's band ≤ 1.15 holds).

**Bitstream:** caplifive_ids55-native-tagcache-mlp_b98f5a517 (capstone-ariane b98f5a517 = 79cf8de17 + an 8-entry
pending tag-write list in wt_axi_adapter: a marked-page load miss's tag read waits only for outstanding tag writes to
its own tag word, where the base drained every outstanding tag write and held the adapter, I$ fills included). Sealed
by the synth lane on leffe: sha256 5e8421a567a60848d3c12bcddb449055f6af8798b16abc2df41c89e8ed21596c (checked after
the copy), WNS −13.738 (base −14.789), 179,661 LUT, router congestion 5, 0 routing errors. The rtl lane's simulation
could not show a saving (the testbench memory answers one transaction at a time, so no tag write is outstanding at a
tag read without the sim-only B-hold switch); the board is the first instrument that can.

**Kit bkit58-k2n-native-tagcache-mlp** = bkit56 with only the bitstream changed, so each reading pairs with bkit56's
(same kernel tree linux-board-k2n-native, firmware, command line, no cache). **Runs, board 2** after its running chain
(x264): imagick, nab, mcf, gcc train01; the native smoke gates every boot as usual.

**Predictions** (against bkit56: imagick 13.75 s, nab 747.14 s, mcf board 1's reading, gcc train01 3,409.84 s):
imagick and nab within ±1.5 %; mcf and gcc train01 0 to −5 %. **Decision rule:** mlp replaces tagcache for the final
table only if mcf or gcc train01 improves by more than 2 % and no row regresses by more than 1.5 %; otherwise the table
stays on cap-ids55-native-tagcache and this run is reported as a negative result.

## Addendum 77 (2026-11-13 07:55, before it runs): Capstone native mcf 1.569; a footprint-only control for mcf on riscv-base

**Reading:** Capstone mcf test, native, cap-ids55-native-tagcache, K2N, no cache: 8,076.96 s, specdiff 2/2 PASS;
riscv-base 5,146.72 s; ratio 1.569 (Addendum 73's band 1.4-1.6 holds). Native counters over the command: fault
escapes +46,494, k1 +21,897, service +1,402.

**Twin instruction profile** (capmix3, the mcf command's window, specdiff PASS; perf/mcfprof): native user 26.80 G,
kernel 2.26 G, firmware 0.23 G = 29.29 G; riscv-base (rv-fair, kit-profq kernel) user 27.32 G, kernel 0.74 G,
firmware 0.08 G = 28.14 G. Instructions give 1.041; the program itself executes 2 % FEWER instructions natively. So
mcf's gap on silicon is cycles per instruction, not instructions.

**Diagnostic (not a table row): riscv-base mcf with capability-sized pointer slots.** perf/mcfpad: the plain-rv64f
build of mcf (same compiler, wrapper and flags, replayed from its make.out; the unmodified replay reproduces the stage's
binary byte for byte, fb555086f151) with every pointer member of struct node, struct arc and BASKET in a 16-byte,
16-aligned slot (a union with char[16]), which gives Capstone's layout: arc 72 -> 112 B, node 176 B, basket 48 B
(static asserts). Remaining differences from the Capstone build: pointer arrays outside these structs (the basket
pointer array) keep 8-byte slots. Twin: specdiff 2/2 PASS, 28.73 G instructions (+1.9 %). Board 1, riscv-base,
bkit54-k2n-rv, after the riscv-base cache chain (Addendum 75).

**What it decides:** how much of mcf's 1.569 is data footprint (which only a layout change or a smaller capability
could remove) and how much is capability- or tag-specific memory traffic (the hardware path: tag reads on marked-page
misses, the tag-write list of Addendum 76). **Prediction:** padded / unpadded riscv-base mcf = 1.15-1.35. Above 1.40
means footprint explains nearly all of mcf's gap; below 1.10 means it is the tag or capability path.

## Addendum 78 (2026-11-13 08:30): RETRACTION of the "firmware" columns of Addenda 75 and 77

Addenda 75 and 77 split the twin instruction profiles into user, kernel and "firmware", counting every PC in
[0x8000_0000, 0x8100_0000) as firmware. That range also holds the kernel's text at its physical addresses, which the
windows execute (kernel memset, found by the compiler lane, which retracted the same split in its imagick profile).
Re-split with OpenSBI as [0x8000_0000, 0x8020_0000) only:

| window | OpenSBI | kernel at physical PCs | kernel (virtual) | user |
|---|---|---|---|---|
| gcc train01 native, no cache | 0.001 G | 0.174 G | 1.831 G | 12.420 G |
| gcc train01 native, cache | 0.001 G | 0.041 G | 0.432 G | 12.585 G |
| gcc train01 riscv-base | 0.001 G | 0.078 G | 0.823 G | 10.779 G |
| mcf native, no cache | 0.002 G | 0.226 G | 2.264 G | 26.800 G |
| mcf riscv-base | 0.002 G | 0.076 G | 0.741 G | 27.323 G |

So the "firmware 0.18 G / 0.04 G / 0.08 G" (gcc) and "0.23 G / 0.08 G" (mcf) were kernel work; real M-mode firmware is
about 0.001-0.002 G in every window. Totals, user counts and the instruction ratios (gcc 1.235 / 1.118, mcf 1.041) are
unchanged. The twin never modelled the firmware's CSCHECK emulation, so nothing here measured it; the board's own
reading (Addendum 79) prices it.

## Addendum 79 (2026-11-13 08:35): readings — Capstone perl makerand; imagick on bkit57 with the first silicon cycle counters

- Capstone perl makerand (native, cap-ids55-native-tagcache, bkit56, no cache): 418.64 s, specdiff PASS (riscv-base
  running on board 3).
- **imagick on bkit57** (Addendum 75's d): 13.68 s, specdiff PASS (bkit56: 13.75 s; ratio 1.215 against 11.26 s).
  **Addendum 75's prediction 13.15-13.55 s FAILS:** the aligned kernel buffers are worth -0.5 %, not 2-4 %.
- Counters over the command, 25 MHz core: checks 2,977 = the emu_fingerprints delta (prediction holds); check_cycles
  4.06 M = 1,364 cycles per emulated check (inside the informational 1,000-20,000), 0.16 s = 1.2 % of the command;
  runs 518, run_cycles 224.5 M (8.98 s, 65.6 %); between_cycles 73.6 M (2.95 s, 21.5 %, 5.7 ms per escape);
  1.75 s (12.8 %) lies before the first entry and after the last escape (the task's exec, setup and teardown).
- With the compiler lane's twin phase split (a null run, imagick -version, in the same window: native 6.9 M
  instructions, riscv-base 9.3 M), the escape phase is the arm-specific part: native kernel about 14.5 M instructions
  over 518 escapes against riscv-base's 6.45 M, at about CPI 5 on silicon; the program runs at CPI about 2.3. So
  imagick's gap is escape handling (and the kernel's page work behind it), not the firmware emulation and not the
  program's instructions.

## Addendum 80 (2026-11-13 09:25, before it runs): riscv-base's perl cache stage runs makerand only; imagick with the cache

- **Reading:** imagick on bkit57 with the mapping cache (Addendum 75's e): 13.30 s, specdiff PASS (no cache, same kernel:
  13.68 s). Escapes 439 (no cache 518), between_cycles 60.1 M (no cache 73.6 M), run_cycles 223.4 M (224.5 M). Its
  ratio waits for riscv-base's imagick with the cache.
- **Change before it runs:** riscv-base's perl stage with the cache (Board 1, Addendum 74's chain) is replaced by a
  copy that runs makerand.pl only (perf/fair/stages/500.perlbench_r-rv-fair2-mk: the test.pl command, its in-band
  files and its host validation removed, nothing else). Addendum 74 compares perl on makerand, so test.pl was board
  time that nothing reads: a riscv-base perl test takes about 3 h 40 min with test.pl, minutes without. QEMU check of
  the copy: 1/1 commands, specdiff 1/1 PASS, 2.78 G instructions.
- **Recorded lapse:** board 3's riscv-base perl without the cache (running since 07:30) still runs test.pl; Addendum
  74 filtered the native stage and not this one. It is left to finish (stopping it would lose its makerand reading and
  move the perl row behind x264); its test.pl reading is not used.

## Addendum 81 (2026-11-13 09:45, before it runs): every Capstone run moves to the best sealed configuration; the table becomes the mapping-cache table on both arms

**Lead's instruction (13 Nov):** one Capstone bitstream carries ALL optimisations and best practices, it is option B
(55-bit lifetime ids) + native, and every experiment runs on it, never on a weaker bitstream. The rtl lane is defining
and building that bitstream (every optimisation and needed fix not yet in b98f5a517, on the 0ef470559 line). Until it
is sealed, the best sealed one is **caplifive_ids55-native-tagcache-mlp_b98f5a517** (0ef470559 option B + native +
R-70 -> 79cf8de17 tag page bitmap + tag-word cache -> b98f5a517 pending tag-write list; C.5a's cached verdict and
stage-B's DYN handshake trim are inside; every full-build key PASS).

**Configuration of the table from here:** Capstone = b98f5a517, kit bkit59-k2n-native-tagcache-mlp-ioalign (bkit57's
kernel: aligned kernel I/O buffers, one check per predicted chunk, cycle counters), native route, the mapping cache
(nat-fair-cache); riscv-base = caplifive_rvbase_d31345cf3, K2N, the mapping cache (plain-rv64fr). One configuration
for both arms, as Addendum 74 intended: the cache is the better setting for Capstone (gcc train01 -12 %, imagick
-2.8 % on bkit57) and its riscv-base counterpart is the same reuse policy. The no-cache readings (Addenda 73-79) stay as
recorded, as the earlier configuration.

**Stopped (on the weaker configuration, or no longer read by the table):**
- board 2: Capstone x264 on cap-ids55-native-tagcache, bkit56, no cache (running since 06:33), and the waiting
  Addendum 76 A/B chain (bkit58). Addendum 76's A/B is withdrawn before it ran: every reading from here is on the mlp
  bitstream; the tagcache readings already taken remain its comparison points.
- board 3: riscv-base x264 without the cache (not started); board 3's chain is ended after its running riscv-base
  perl (no cache), which completes and is read.
**Runs:**
- board 2 (g): the eight Capstone rows on the best configuration, short first: imagick, nab, lbm, perl makerand, gcc
  train01, xz, mcf, x264. Native cache stages for nab, lbm, xz, mcf, x264 built today with the same converter as the
  others; each differs from its no-cache stage only in the program image and the plan's image path.
- board 3 (h): riscv-base with the cache: nab, lbm, xz, mcf, x264.
- board 1 (unchanged): Capstone perl with the cache on bkit57 (running; its imagick and gcc already read), then
  riscv-base with the cache: imagick, gcc train01, perl makerand (Addendum 80), then the padded-mcf diagnostic.
**Predictions** (mlp + cache against tagcache, same kernel where it exists): imagick 13.0-13.4 s; gcc train01
2,850-3,050 s; every other Capstone row within -5 % to +1.5 % of its no-cache bkit56 reading; specdiff PASS everywhere.

## Addendum 82 (2026-11-13 11:35, before it runs): a fairness defect in riscv-base's mapping cache; readings so far; a board-vs-bitstream cross for perl

**FAIRNESS DEFECT, against riscv-base (found by the compiler lane, read in perf/fair/musl-src-reuse):** riscv-base's
plain_reuse cache has never reused an individually mmapped chunk. malloc.c:309 maps a large chunk with the exact
request (`__plain_map(n + IB + UNIT)`), free.c:25 releases `maplen*4096`, the cache stores the page-rounded length and
looks up the exact one, so the two never match. Capstone's mapping cache rounds both sides (heap_round) and does reuse
them. So every "mapping cache on both arms" reading since Addendum 74 gave Capstone a reuse riscv-base did not get,
wherever large blocks are freed and re-made. QEMU, riscv-base mcf test (kit-profq, specdiff 2/2 in every variant):
as built, 0 hits on mcf's 8 arcs_pointer_sorted mappings, pgfault 132,539; with rounding and eviction, 6 hits,
pgfault 85,654. The ratios with the cache listed below are therefore biased in Capstone's favour where large mappings
recur (mcf above all, gcc to a degree), and none of them is final.

**Fix, and the next configuration (both arms rebuilt, one step):** riscv-base gets the rounding fix (a baseline bug,
not a policy) plus the eviction rule (musl perf/plainevict/musl-rv64gc-reuse-round-evict, wrapper
plain-cc-fairr-round-evict); Capstone gets the same eviction rule plus in-place growth of large reallocs (compiler
lane nat-realloc-inplace f097dd782aae and nat-realloc-evict 3e70344d8c3f, musl patch 0011; riscv-base already grows
large reallocs in place through mremap). Twin, Capstone mcf: kernel 2.484 G / 392,471 pages (cache) -> 1.596 G /
248,406 (+ in-place realloc) -> 0.916 G / 138,140 (+ eviction), outputs byte-identical. Both image sets are rebuilt
and every row of both arms is re-run on them; a separate addendum pre-registers that run.

**Readings since Addendum 81** (specdiff PASS everywhere; the cache readings carry the defect above):
- Capstone, mlp + bkit59 + cache (board 2): imagick 13.30 s, nab 745.19 s, lbm 1,072.63 s, perl makerand 409.41 s.
- Capstone, tagcache + bkit57 + cache (board 1): perl makerand 400.99 s.
- riscv-base + cache (board 1): imagick 11.02 s, gcc train01 1,721.71 s, perl makerand 254.34 s.
- With the cache on both arms: imagick 1.207; gcc train01 (tagcache) 1.742; perl makerand 1.610 (mlp) / 1.577
  (tagcache). Addendum 81's predictions for imagick, nab, lbm and perl hold.

**The perl pair:** mlp 409.41 s against tagcache 400.99 s with the same kernel and images, +2.1 %, all in the domain
(run_cycles 10.095 G against 9.886 G; escape and check counts identical). They ran on DIFFERENT boards (mlp on board 2,
tagcache on board 1), and the two builds differ in RTL only in wt_axi_adapter.sv (rtl lane, by git diff), so it is
either the tag-write list (an rtl-lane simulation A/B is queued) or the board. **Run, board 1** after its running
diagnostic: mlp + bkit59 + cache, perl makerand then imagick. Within 1 % of 409.41 s means the bitstream; within 1 %
of 400.99 s means the board, and then every cross-board ratio carries that effect and the table needs same-board
pairs. No outcome is favoured.

## Addendum 83 (2026-11-13 11:25, before it runs): riscv-base on the fixed mapping cache, all eight rows, board 3

**Reading:** riscv-base perl makerand without the cache (board 3): 254.80 s, specdiff PASS (its test.pl, 12,581 s, is
not used; Addendum 80).

**Build:** label plain-rv64fre = plain-rv64fr.cfg with only the label changed; compiler wrapper
perf/plainevict/plain-cc-fairr-round-evict = perf/fair/plain-cc-fairr with only the musl changed, to
perf/plainevict/musl-rv64gc-reuse-round-evict (plain_reuse.c: __plain_map rounds the mapped length to pages as free()
does, and __plain_unmap evicts cached mappings from the front until a mapping that fits the 64 MiB bound by itself
fits; the Capstone arm's rule). runcpu rc 0; the new __plain_map and __plain_unmap are present in the binaries
(objdump: 116 -> 120 and 33 -> 111 instructions, a second munmap call in the eviction loop). Stages
perf/fair/stages/*-rv-fair3 (perf/fair/mkstages3.py): copies of the *-rv-fair2 stages (perl from the makerand-only
-mk copy, gcc from train01) whose only differences are the executables and their plan hashes (checked by diff).
QEMU (kit-profq) on each before the board: specdiff PASS (x264 not run on QEMU: its board validation is by hash and
imagevalidate as before).

**Run, board 3** (bkit54-k2n-rv, K2N, riscv-base bitstream): imagick, nab, lbm, perl makerand, gcc train01, xz, mcf,
x264. The Capstone arm on its matching allocator (in-place realloc + eviction) follows when the compiler lane's images
pass their twins; that is pre-registered separately.

**Predictions** (against plain-rv64fr's readings where they exist): imagick, nab, lbm, perl within ±1.5 %; gcc
train01 1,650-1,725 s; mcf 2-8 % faster than without any cache (5,146.72 s); xz and x264 within ±2 % of their
no-cache readings.

## Addendum 84 (2026-11-13 11:45, before it runs): the Capstone arm on its matching allocator, all eight rows, board 2

**Images (compiler lane):** llvm-capstone nat-realloc-evict 3e70344d8c3f = nat-fair-cache + nat-realloc-inplace
f097dd782aae (musl patch 0011: a growing mmapped chunk stays in place when its mapping has room, otherwise moves to a
mapping with room for twice the new size) + mr_make_room (a mapping that fits the 64 MiB bound by itself evicts older
cached mappings until it fits: the rule riscv-base's plain-rv64fre now has, Addendum 83). Results and the riscv-base
diffs at 6aa944e5ee3b (realloc-grow/RESULTS.md, realloc-grow/riscv-base/plain-reuse-round-evict.diff). Exes
perf/natevict/exes, built with nat-fair-cache's replay and flags. Twins (outputs byte-identical to the cache and
in-place arms): mcf kernel -63.1 %, gcc train01 -3.2 % (service escapes 2,600 -> 2,303), imagick +0.3 %, xz and perl
0.0 %; lbm and nab identical outputs; x264's twin still running (its board row runs last).

**Stages** perf/natstage/board-evict/*: built with the same converter and kit script as board-cache; each differs from
its board-cache stage only in the program image and the plan's image path (diff -rq); perl carries the same
makerand-only edit (Addendum 74/80).

**Run, board 2**, after its running gcc train01 (mapping cache A, mlp): mlp + bkit59 + arm C, imagick, nab, lbm, perl
makerand, gcc train01, xz, mcf, x264. Board 2's remaining A stages (xz, mcf, x264) are dropped: superseded by C and
never part of a fair pair (Addendum 82).

**Predictions** (against the A readings on mlp where they exist, else against no-cache bkit56): imagick, nab, lbm,
perl, xz within ±1.5 %; gcc train01 0 to -3 % against A on mlp (lands within the hour); mcf 3-10 % faster than its
no-cache 8,076.96 s (twin kernel -63 % of 2.5 G instructions, at the program's CPI); x264 within ±2 % of no cache.

## Addendum 85 (2026-11-13 13:00, before it runs): readings; board 2's arm-C queue split with board 1; riscv-base xz not re-run

**Readings** (specdiff PASS everywhere):
- Arm C (mlp + bkit59 + matching allocator, board 2): imagick 13.29 s, nab 741.97 s. Against riscv-base plain-rv64fre
  (board 3): imagick 13.29 / 11.08 = 1.1995; nab 741.97 / 679.92 = 1.091. riscv-base plain-rv64fre also: lbm
  1,077.50 s, perl makerand 257.90 s (Addendum 83's ±1.5 % bands hold).
- Padded-mcf diagnostic (Addendum 77, riscv-base, board 1): 5,854.28 s against unpadded 5,146.72 s (board 3) = 1.137,
  just under the 1.15-1.35 band. Data footprint is ~24 % of Capstone's no-cache mcf gap (1.569); the rest is
  capability- and tag-specific, so the DYN bypass is the right lever for mcf too.
- Board-vs-bitstream cross (Addendum 82, cache A): perl makerand mlp board 1 405.56 s (mlp board 2 409.41, tagcache
  board 1 400.99): board term -0.9 %, bitstream term +1.1 %; imagick mlp board 1 13.31 s (board 2 13.30). Both terms
  are at the ~1 % noise level of single readings; with gcc train01 at mlp -0.94 % and the rtl lane's drain-vs-list
  simulation (8 of 8 pairs equal or faster), mlp stays the base. Single readings carry about ±1 %.
- Reader fix: read-all.sh read the arm-C runs against the no-cache stages' plans; it now uses stages-evict (whose perl
  carries the makerand-only edit). Re-read: imagick and nab unchanged.

**riscv-base xz is not re-run on the fixed cache:** QEMU instret over the 12 commands is 59.618 G without the cache,
59.631 G with the old cache and 59.576 G with the fixed one (specdiff 12/12 each), so the change does not move xz; its
no-cache board reading 4,993.81 s stands as the pair for arm C (lead: no riscv-base re-run without a strong reason).
gcc train01, mcf and x264 stay on board 3 (mcf and gcc are where the fix matters; x264 has no K2N reading at all).

**Split of board 2's arm-C queue** (lead: problem rows first): board 2 ends its chain after the running lbm and runs
perl makerand, gcc train01; board 1 (idle, mlp resident) runs mcf, xz, x264. Same kit, images and stages as Addendum 84.

## Addendum 86 (2026-11-13 13:20, before it runs): LEAD'S RULING — riscv-base is stock CVA6 with stock software; the riscv-base mapping cache is withdrawn

**Ruling (the lead, 13 Nov):** the baseline is the default CVA6 processor and its default software; optimising it is
not this project's task — only the Capstone arm is optimised. So the mapping cache that Addendum 67/74 added to
riscv-base's musl (plain_reuse, plain-rv64fr) and Addendum 83's fixed version (plain-rv64fre) are withdrawn from the
baseline. riscv-base is plain-rv64f: stock musl malloc, the same llvm-c4 compiler and flags as Capstone's arm
(-fno-pie and static, as before: those are equal-conditions settings, not optimisations), the same K2N kernel, the
stock-CVA6 bitstream caplifive_rvbase_d31345cf3. The Capstone arm keeps its own allocator work (mapping cache,
eviction, in-place realloc: Capstone's runtime, Addendum 84).

**Consequences:**
- riscv-base readings of the table are the no-cache ones on K2N: imagick 11.26 s, nab 675.63, lbm 1,077.88, gcc
  train01 1,879.48, mcf 5,146.72, xz 4,993.81, perl makerand 254.80 (Addenda 72-83). The plain-rv64fr/-fre readings
  (Addenda 74-85) are kept as recorded and no longer enter any ratio.
- Board 3's plain-rv64fre chain is stopped (its mcf mid-run, booking released). Addendum 82's "fairness defect"
  concerned that withdrawn cache only.
- The one missing riscv-base row, x264 without the cache (plain-rv64f, perf/fair/stages/525.x264_r-rv-fair), runs on
  board 3 now. Prediction: none recorded before on K2N; specdiff/hash PASS.
- Arm C ratios so far against the stock baseline: imagick 13.29 / 11.26 = 1.180; nab 741.97 / 675.63 = 1.098; lbm
  1,083.68 / 1,077.88 = 1.005.

## Addendum 87 (2026-11-13 13:35, before it runs): the lead's CLARIFICATION of Addendum 86 — same allocator on both arms, the baseline's RTL only

**Clarification (the lead, 13 Nov, after Addendum 86):** the allocator must be the SAME on both arms; what is the
baseline's own is the RTL (the bitstream). So Addendum 86's withdrawal of riscv-base's mapping cache is itself
withdrawn: riscv-base runs plain-rv64fre (Addendum 83: the mapping cache with the rounding fix and eviction, the same
policy as the Capstone arm's allocator, whose large reallocs riscv-base's musl already grows in place through mremap)
on the stock-CVA6 bitstream caplifive_rvbase_d31345cf3. Everything else stays equal on both arms (compiler and flags,
static linking, K2N kernel, inputs, measurement).

**Baseline readings of the table** (plain-rv64fre, board 3, specdiff PASS): imagick 11.08 s, nab 679.92, lbm 1,077.50,
perl makerand 257.90, gcc train01 1,722.35; xz 4,993.81 (no-cache reading; QEMU shows the cache does not move xz,
Addendum 85). Missing: mcf (stopped mid-run under Addendum 86) and x264.
**Run, board 3:** riscv-base plain-rv64fre mcf, then x264 (the stock-musl x264 started under Addendum 86 is stopped).
Predictions as Addendum 83 (mcf 2-8 % faster than its no-cache 5,146.72 s; x264 within ±2 % of a no-cache reading,
which does not exist on K2N).
**Ratios so far** (arm C on mlp + bkit59 against plain-rv64fre): imagick 1.1995, nab 1.091, lbm 1.006.

## Addendum 88 (2026-11-13 14:30, before it runs): arm C gcc train01; board 2's idle slot takes x264

**Reading:** arm C gcc train01 (mlp + bkit59, board 2): 2,963.19 s, specdiff PASS; against riscv-base plain-rv64fre
1,722.35 s: 1.720 (run_cycles 71.2 G of the command; escapes 18,354 runs; the program's CPI carries the gap).
Arm C so far: imagick 1.1995, nab 1.091, lbm 1.006, perl makerand 1.551, gcc train01 1.720.
**Rebalance:** board 2 finished its arm-C rows and is idle; x264 moves from board 1's list to board 2 now (same kit,
image and stage). Board 1's chain ends after its running mcf; it then runs xz only.

## Addendum 89 (2026-11-13 17:00, before it runs): arm C xz and mcf; board 1 becomes the A/B board, its base run on the all-optimisations bitstream with the new compiler and the HPM kit

**Readings** (arm C, mlp + bkit59, specdiff PASS): mcf 7,605.65 s (board 1) against riscv-base plain-rv64fre 5,125.02
(board 3) = 1.484; xz 5,462.89 s over 12 commands (board 1) against 4,993.81 = 1.094. Arm C, all rows so far:
imagick 1.1995, nab 1.091, lbm 1.006, xz 1.094, perl makerand 1.551, gcc train01 1.720, mcf 1.484 (x264 running).

**Base of the A/B**, board 1 (idle since 16:52):
- bitstream caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay_5f41c486e (sealed: b457b57a44282cd6...; WNS -15.077,
  181,565 LUT; b98f5a517 + R-64/65/66 FP fix + HPM event 23; no cycle change in simulation);
- kit bkit61-k2n-native-best-hpm = bkit60 + native kernel with llvm-capstone native-hpm-counters e517119e12c4
  (hpmcounter3..8 summed over each CSRUNV: hpm3 L1I miss, hpm4 L1D miss, hpm5 DTLB miss, hpm6 mispredict, hpm7 issue
  stall, hpm8 scoreboard full; firmware wrapper 8a39f82d programs them) + command-line parameters
  capstone_vm.dom_grant_root=/tmp/nt/sr dom_null=1 dom_local_tmp=1 (perl's full test; 540-byte CONFIG_CMDLINE, board
  limit 1024);
- images: FR = arm C's SDK built by llvm-c6 (llvm-capstone compiler-global-lo-fold 2f3ff76eba7b, -capstone-fold-global-lo,
  the padded link.ld) with call relaxation (+relax codegen). Twins: outputs byte-identical to arm C on every row but
  x264 (its twin lands about 19:05); perl makerand -6.92 % user instructions, gcc train01 -1.20 %; the QEMU gate
  (runtime/virtual contracts) passes as the control does.
- rows: perl makerand, gcc train01, mcf (problem rows first).
**What it reads:** the FR compiler's effect (against arm C on mlp, same allocator; the bitstreams differ only by the FP
fix and the counter), and the base for the DTLB-64, L1D-64K and DYN-bypass bitstreams, which run on this board with the
same kit and images when sealed. **Predictions:** perl makerand 375-395 s; gcc train01 2,900-2,960 s; mcf within ±1.5 %
of 7,605.65 s; hpm4/hpm5 non-zero and monotone over each command, hpm5 (DTLB misses) on the order of the model's
(mcf ~0.9 G, gcc ~0.2 G, perl ~36 M).
