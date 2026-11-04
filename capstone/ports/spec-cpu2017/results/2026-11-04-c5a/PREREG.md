# C.5a on board 1: first boots, then SPEC test: pre-registration

This file is written and pushed before the flash. The lead's word for the flash: "прошивай если нужно" (4 Nov).

**The bitstream.** `caplifive_vp2c_1fc2cd5aa` is capstone-ariane 1fc2cd5aa, with two changes:
- LDC and STC take the LSU's cached lifetime verdict (the revocation-validity cache);
- the CJALR / PC-tracker soundness fix.

Its record is `docs/plans/2026-11-03-c5a-bitstream.md` (land/rtl-c5a fb1172077b63).
- .bit sha256 `5b7b1edcd9bcd2521bd4bd59107747e268487056f1bfcd96403fd612eeb6d764`. It was extracted from the synth lane's tarball and checked locally.
- WNS is −13.758 ns (b3 −10.047). Bitstreams at −18.5 and −19.0 have run in service.
- Fallback: bitstream 3, `caplifive_vp2c_e7af48457`, sha256 `d6ed97b6…`, present locally.

**When.** After board 1's compiler-evaluation pair on mcf (`505.mcf_r-test-capstone-lifetimes-dev` and `-devmk`), the
last stages that must run on bitstream 3 before the rest of the queue.
- A sentinel line in the chain's list ends the chain there.
- After the C.5a work, board 1 is flashed back to bitstream 3, and the chain restarts on the rest of its list: x264 Plain test, then SPEC train.

**Runner.** `c5a-b1.sh`, committed beside this file.
- Every board call was stubbed in its dry run (`DRYRUN.md`), which took eight branches:
  - success;
  - R-67 not FIXED;
  - the C.5a flash fails;
  - the chain stops somewhere other than the sentinel;
  - a HARD STOP during the hpm kit, and one during the ObjBounds kit;
  - the flash back to bitstream 3 fails;
  - the pre-registration is missing.
- The HARD STOP branch found a defect: SPEC would still have run. It was fixed and re-run.

## Part 1: the kits' first boots (the rtl lane's pre-registrations)

One image per boot, through `run_sup_bare.py`, which stops HARD if the resident name is not C.5a's.

| Order | Kit | Images | Grade | Pre-registered at |
|---|---|---|---|---|
| 1 | R-67 | t1-norevoke, t1-untagged, t1-oob, t1-fire | `r67-grade.py <run> fixed --bitstream 1fc2cd5aa`: FIXED, rc 0 | land/rtl-r67-fixed 63e60355c408 |
| 2 | hpm | hpm-5, hpm-3, hpm-4, hpm-6, hpm-1, hpm-2, hpm-7 | `hpm-grade.py <run> c5a perf/hpm/run-b1-bs3`: AS EXPECTED, cost pattern HOLDS | land/rtl-c5a-kits 6dae10292923 |
| 3 | ObjBounds | objb-pcc-ctrl, objb-stale, objb-stale-w, s1-s7, objb-pcc LAST | `objb-grade.py <run> c5a perf/objb/bare-bs3-b2`: AS EXPECTED, cost pattern HOLDS | land/rtl-c5a-kits 6dae10292923 |

- The graders are the rtl lane's, at those commits.
- Each grader was run in this invocation form on the bitstream 3 runs. All read rc 1:
  - hpm: WRONG BITSTREAM;
  - ObjBounds: COST PATTERN FAILS;
  - R-67: IMAGE OR BITSTREAM MISMATCH.
- **R-67 gates the rest.** Anything but FIXED stops C.5a work, and the reading goes to the lead.
- **SPEC needs all three.** SPEC runs only if all three grade rc 0 and no image hit a HARD STOP.

## Part 2: SPEC test on C.5a

**Kit.** `t3-smoke/bkit17-perf-c5a-b1` is bkit12-perf-b3-b1 with `BOARD_BITSTREAM=caplifive_vp2c_1fc2cd5aa` and SHA256SUMS regenerated. It has the same launcher and module, which spec-board.sh checks by hash. The control is crc.K0 T1.

**Stages.** The 3 Nov board 1 stages, with the same images, steps and twin (census-q34), cheapest first:

| Order | Stage | Arms |
|---|---|---|
| 1 | `stages-board/502.gcc_r-test-capstone-lifetimes` | Full |
| 2 | `stages/538.imagick_r-test` | MapLife, Full |
| 3 | `stages/544.nab_r-test` | MapLife, Full |
| 4 | `stages/519.lbm_r-test` | MapLife, Full |
| 5 | `stages-board/557.xz_r-test-capstone-lifetimes` | Full |
| 6 | `stages-board/505.mcf_r-test-capstone-lifetimes` | Full |

Out root: `spec/board/out-c5a`. There are two references:
- bitstream 3, board 1, 3 Nov (`spec/board/out/`);
- ObjBounds Spatial, board 2 (`spec/board/out-objb/<bench>-test-capstone-spatial`).

**Today's gap** (run commands, in-context cycles):

| Benchmark | Full b3 / Spatial ObjB | MapLife b3 / Spatial ObjB |
|---|---|---|
| lbm | 1.0057 | 1.0054 |
| nab | 1.0646 | 1.0499 |
| imagick | 1.0189 | 1.0381 |
| xz | 1.0389 | |
| mcf | 1.1153 | |
| gcc | 1.0617 | |

**The model.** The QEMU lc-model (qemu-lc 0f271af7befc, prof/outlc/SUMMARY.txt) counts LDC+STC misses in C.5a's cache:
lbm 0.001 %, nab 0.004 %, imagick 0.03 %, xz 0.06 %, gcc 0.17 %, mcf 1.88 %. At 12-13 cycles per miss over ObjBounds,
that is mcf about +0.75 % of its run and every other benchmark under 0.1 %.

**Predictions** (grader `c5a-spec-grade.py`; its controls: the bitstream 3 runs given as C.5a read rc 1, a mock of the
predicted values reads rc 0, no data reads rc 2):

| # | Prediction |
|---|---|
| S1 | The control passes. Every arm is rc 0 and specdiff PASS, its outputs equal the twin's, and its bench line is fresh. instret equals the bitstream 3 run within 0.1 %: the same images. |
| S2 | C.5a / Spatial ObjB cycles lie in 1 + model ± 0.01. That is 0.990-1.010 for every arm, and 0.998-1.018 for mcf Full. The ±1 % allows for board 1 against board 2 and day against day, which are not measured. |
| S3 | Full on C.5a takes fewer cycles than Full on bitstream 3 for nab, imagick, xz, mcf and gcc, the five whose gap today is ≥ 1.8 %. lbm is reported. |

**What would refute what.**
- **S1 fails.** C.5a runs these programs differently. Possible causes are a defect or a timing failure at −13.8 ns. No cycle verdict stands, and the reading goes to the rtl lane and the lead.
- **S2 fails high.** Lifetime checks still cost more than the model's misses: C.5a's misses are more frequent or dearer than modelled, or C.5a costs something elsewhere. S3 then says how much of the gap is left.
- **S2 fails low.** C.5a is faster than ObjBounds. Board-to-board drift is the first suspect.

**Limits.**
- N = 1 per arm.
- S2 compares two boards.
- lbm's gap today (0.6 %) is inside the band, so lbm cannot confirm or refute S2.
- The MapLife arms run the Spatial images on a bitstream with the lifetime check; they are predicted like Full.

## Addendum 1 (4 Nov 13:40, before any board call): the slot-bare kit on bitstream 3 and on C.5a

The rtl lane's slot-bare kit is added: llvm-capstone land/rtl-slot-kit e1a835aa466e, `capstone/tests/rtl-smoke/slot-bare-2026-11-04`. It runs nine arms of the STC→LDC pair and its controls, slot-9 first.
- **Its predictions are the rtl lane's** (that PREREG; grader `grade/slot-grade.py <run> <bs3|c5a>`):
  - instret, value, traps and cause exact;
  - P1: slot-1..4 equal within 2 %;
  - P2: slot-5 > slot-6.
- **Where it runs.** On board 1, once on bitstream 3 before the C.5a flash (the board is still bitstream 3 when the chain stops), and once on C.5a after the hpm and ObjBounds kits.
- **What it gates.** Nothing, except that a HARD STOP (resident mismatch) on C.5a blocks SPEC.
- **Why.** Its per-op costs on silicon (ldc, stc, the pair, sd+ld) are what the DYN-unit work chooses by. Simulation on C.5a: pair 8.42 cycles; silicon saves 23-36 per removed pair.

The runner was stopped by hand before any board call and relaunched with this step. Its dry run (S1 success, S2 R-67 not FIXED, S3 HARD STOP in the slot kit) is in DRYRUN.md.

## Addendum 2 (4 Nov 15:50): the C.5a session waits for the fault-ahead run on board 1

The fault-ahead run moved to board 1 (fault-ahead PREREG Addendum 4) and ranks above C.5a. `c5a-b1.sh` was stopped by hand while still waiting; it had made no board call.
- `fa-b1.sh` launches it when the fault-ahead run ends, about 9 h after the mcf pair.
- Its sentinel, its order and its predictions are unchanged. It finds the chain's pid gone and the sentinel stop in the chain log, and proceeds.

## Addendum 3 (4 Nov, before any board call)

After the bitstream 3 slot run and before the flash, c5a-b1.sh now runs the two counter boots of hpm PREREG Addendum 2 (imagick dev and devmk on bkit16-hpm-b3-b1). They are informational and gate nothing. Dry run S4-hpm.

## Addendum 4 (4 Nov 16:55, before any board call): the session moves to board 2

The lead ruled that optimisation ranks above the paper's train. Board 2's x264 train Full was stopped, together with the wrappers that queue train. Board 2 is on bitstream 3, read back at 15:23. It runs this session now with `c5a-b2.sh`, which is `c5a-b1.sh` with these changes:
- **Board.** Board 2's console, busy check and hub bookings.
- **Kit.** `bkit17-perf-c5a-b2` is bkit12-perf-b3-b2 with `BOARD_BITSTREAM=caplifive_vp2c_1fc2cd5aa`; spec-board.sh checks its launcher and module by hash.
- **Output.** perf/c5a-b2; SPEC into spec/board/out-c5a-b2.
- **Dropped:** the chain wait and sentinel; the counter boots of Addendum 3, because no bitstream-3 counter kit exists for board 2; and the chain restart. Board 2 ends back on bitstream 3, idle.
- **Unchanged:** the order (slot kit on bs3, flash, R-67 gate, hpm and ObjBounds kits, slot kit on C.5a, then SPEC only if all three read rc 0 and no HARD STOP), the graders and their reference dirs, and S1-S3.

**Reference boards.** S2 (C.5a / Spatial ObjB) is now board 2 against board 2, the same board. S3 (C.5a / bs3 Full) is board 2 against board 1's 3 Nov run.

Dry run B2-ok (40 image runs, all on board 2; bookings and SPEC chain on board 2): perf/c5a-b2/dryrun.

## Addendum 5 (4 Nov 17:50, after the hpm kit ran and before its grade and the SPEC gate): SPEC is not blocked by the hpm kit's one-miss FINDING

**Board 2 so far** (`c5a-b2.sh` relaunched 16:57):
- The slot kit on bitstream 3 graded AS EXPECTED, with the same cycles as board 1.
- C.5a was flashed and read back at 17:34.
- **R-67 graded FIXED.**

**The hpm kit.** The rtl lane graded it with 5b6fb6a4ec9b against run-b1-bs3 and read FINDING.
- **What fails.** l1d_miss, an exact field, reads 1 where simulation says 0 on hpm-5, 1, 2, 3 and 4. On hpm-7 it reads 257 against 256.
- **Everything else matches.** l1d_acc equals the simulation. Bitstream 3 and ObjBounds read the predicted value, so the extra miss is new on C.5a (N = 1).
- **The predicted effect is there.** The lifetime query's stall reads 509 / 254 (ObjBounds 511 / 256, bitstream 3 1,530 / 1,530).
- **So the session skips SPEC.** Under Part 1's rule, hpm must read rc 0, and this reading will make c5a-b2.sh skip SPEC.

**The amendment, written before c5a-b2.sh grades hpm:**
- **Why it is not a blocker.** One extra L1D miss per run is a counter discrepancy, not a correctness or cost failure. The SPEC gate exists to keep SPEC off a C.5a that misbehaves, and SPEC's own S1 (control, outputs, instret) guards correctness.
- **The follow-up runner.** `c5a-spec-b2.sh` waits for c5a-b2.sh to end and runs SPEC on C.5a only if all of these hold:
  - R-67 read FIXED;
  - the ObjBounds kit graded rc 0;
  - the hpm grade reads neither WRONG BITSTREAM nor NO DATA;
  - no image hit a HARD STOP;
  - c5a-b2.sh did not run SPEC itself.
- **What it does.** It re-flashes C.5a (sealed sha, read back), runs the same stage list into out-c5a-b2, grades S1-S3 unchanged, and flashes bitstream 3 back.
- **The FINDING stays a FINDING.** A second C.5a kit session on board 1 later is to separate the bitstream from this session's board state.

Dry runs CS-ok, CS-noR67, CS-hard, CS-ok2, CS-objbfail and CS-ranspec are in perf/c5a-b2/dryrun.
