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
