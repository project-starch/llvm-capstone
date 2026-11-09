# Native QUICK tests on e5-native (option A): gcc, imagick, nab, lbm at test size: bundle and pre-registration

MEASUREMENT ONLY. The collaborator's direction: decide between native and the module route tonight, from quick
test-size board runs. The order is gcc, imagick, nab, lbm. Written before any board run.

## What runs

- **Bitstream:** caplifive_e5native_14b0a57df (sha256 b6698b7c…a475), through perf/e5ngate/e5n-gate.sh.
- **Kit:** bkit38-native-mcf, unchanged:
  - 2^18 table;
  - firmware capstone-sbi e04ecb8 with CAPSTONE_CSCHECK_EMU, whose RTL arm passed on 14b0a57df;
  - the K-min kernel tree linux-board-c5a.
- **Driver:** perf/native-qt/qt-run.sh <board> <prereg>.
  - It runs e5n-gate.sh, then perf/native-qt/nt-run-qt.sh once per stage, each with its own NT_O: a smoke boot,
    then the stage boot.
  - nt-run-qt.sh is perf/native-train/nt-run.sh, except that it also accepts a pre-registration on
    origin/native/c5a-ids55.
- **Stages:** native01/board-stages/qt-{gcc,imagick,nab,lbm}, each with a .SHA256SUMS that qt-run.sh checks.
  - They were built by the kit's make-train-stage.sh from stages-native/<bench>-test-native-devmk.
  - Those come from make-native-stage.py on the module route's own devmk test stages, stages-eval-b1/<bench>-test-devmk.
    The two-arm stages were first cut to the lifetimes arm by native01/spec/one-arm-stage.py.
- **Executables:** the module exes' make lines, the capstone-cc lines of tree-boarddevmk's make.out (module exes:
  gcc d591641944f63655, imagick f50dc465183e4f1f + imagevalidate 8ecf59b59b9bc166, nab 092ff45c0c519283,
  lbm f0269eea1b555bd9), replayed by native01/spec/build-bench.sh with the native-IO SDK native01/sdk-native-c1.
  That is the SDK of the native mcf train exe. Then they were stripped as make-stage.py strips (llvm-objcopy
  --strip-all, keeping .capstone_virtual and .capstone_domreq).

| stage | SHA256SUMS sha256 | native exe(s) sha256 (first 16) |
|---|---|---|
| qt-gcc | b3bcbdc6ccea0ec1 | cpugcc_r df93c433bb1a5418 |
| qt-imagick | 3476b3b29141ac50 | imagevalidate_538 663ed20608ef721c, imagick_r 8601cb8942d9d2e6 |
| qt-nab | 87f4cacaad421947 | nab_r 4275d4b599c5d7ab |
| qt-lbm | ddf07d50dab9001f | lbm_r 6e8d8b4fc511009e |

- **One edit beyond make-native-stage.py:** imagick's run.sh copies stage/ref into the grant root, and imagevalidate
  reads its reference from there. The first twin of the generated stage failed imagevalidate with
  "'/mnt/vm/ref/031fce06-test_expected.tga': Could not open for read". The native grant confines paths, and ref/
  sat outside it.
- **Twin20 smoke:** qemu20, images-flat20d, twin20-936-c5a.dtb, 936M, 2^18 table, direct arm (native01/spec/qt-twin.sh):
  - gcc: rc 0, specdiff 1/1;
  - imagick: 2/2 rc 0, specdiff 1/1;
  - lbm: rc 0, specdiff 1/1;
  - nab: running at this commit;
  - adopt_kind_refused 0 in every run.
- **The twins ran the unstripped exes.** Stripping removes only symbols: nab's .text is byte-identical before and
  after.

## Predictions (per stage)

1. The gate prints `E0 GATE PASS: board <B> stays on caplifive_e5native_14b0a57df`, or nothing runs.
2. The smoke boot's classify.py reads PASS (the kit's accounting waiver).
3. Every command rc 0, and the host specdiff PASSES on its in-band output.
4. Output hashes equal the twin's:
   - gcc t1.opts-O3_-finline-limit_50000.s 58111208509dd871;
   - lbm 100_100_130_cf_a.of 18e20725a7638f36 and lbm.out 7ab5463219d503a3;
   - imagick test_output.tga as the twin's results;
   - nab as its twin.
5. emu_fingerprints > 0 and adopt_kind_refused 0.
6. Run time is the result. The module route's devmk readings for comparison are imagick 22.6 s, nab 991.1 s and
   lbm 1,811.0 s, the best Capstone test runs on C.5a (results/2026-11-07-plainreuse); gcc test is about 26 s.
   Native runs on e5-native, a different bitstream. Its ratio to Plain uses the same Plain-with-cache test
   denominators.

## Addendum 1 (before any board run): imagick, nab and lbm are unstripped, as the module route's exes are

I compared the module stages' exes with tree-boarddevmk's built exes:
- nab 092ff45c, lbm f0269eea, imagick f50dc465 and xz 0d8878ad equal the tree's UNSTRIPPED exes byte for byte;
- only gcc's stage exe (d5916419) is the stripped one.

So the native stages now carry the same treatment:
- imagick, nab and lbm unstripped, which are exactly the exes their twins ran;
- gcc stripped. Its twin ran the unstripped build of the same link.

The table above is superseded by:

| stage | SHA256SUMS sha256 | native exe(s) sha256 (first 16) |
|---|---|---|
| qt-gcc | b3bcbdc6ccea0ec1 | cpugcc_r df93c433bb1a5418 (stripped) |
| qt-imagick | f2502b6fcbdaf58b | imagevalidate_538 6e5613cc944bb95d, imagick_r f008168e93193db3 |
| qt-nab | 6669b525e9513708 | nab_r 3007ce4c2240055b |
| qt-lbm | 410af1b21f3049b3 | lbm_r ed43d5cc37d18396 |

Twin20 smokes, all four: rc 0, specdiff 1/1, adopt_kind_refused 0 (imagick 2/2 commands).

## Addendum 2 (before any board run): the hand-off command, two boots of stages

perf/train/qt-board2.sh runs e5n-gate.sh 2 and then perf/train/qt-native.cmd with QT_GATE_LOG. That file calls
perf/native-train/nt-run.sh twice, each time with NT_WAITLOG=$QT_GATE_LOG and kit bkit38-native-mcf:
- qt-gcc (SHA256SUMS sha256 b3bcbdc6ccea0ec1): 502.gcc_r test;
- qt-inl (SHA256SUMS sha256 02686954ecefd86f): 538.imagick_r, 544.nab_r and 519.lbm_r test in one boot,
  in that order.

qt-inl's train/, train-nab/ and train-lbm/ are the stage directories of qt-imagick, qt-nab and qt-lbm, byte for
byte. Its smoke part equals qt-imagick's.

Why two stages, not one: all four in one initramfs would sit at the kit's 26,000,000-byte firmware cap. qt-gcc
alone is the module gcc test's size.

Exit 0 only if both boots ran and every command ended rc 0 with its @@SPEC-DONE (qt-gcc: 1 command, 1 done;
qt-inl: 4 commands, 3 done). The predictions of the main section apply per benchmark.
