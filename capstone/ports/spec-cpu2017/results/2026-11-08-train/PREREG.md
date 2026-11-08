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
