# Spatial-only core (steps A + B): full build, board lane pre-registration

Status: pre-registered 2026-11-20, before the launch. The keys are the rtl lane's (land/rtl-spatial-prereg 9dacb7d2658e,
`capstone/docs/plans/2026-11-20-spatial-only-core.md`), adopted unchanged. Brief: programme line A row 2.

**Tree, read at source.** capstone-ariane spatial-only-b 1f1ed11f9; P1 d83d3a633 and step A cf7a00ce6 are both
ancestors.
- Against P1, `core/` changes in seven files, +149 lines with none removed.
- Against step A, it adds `core/decoder.sv` +9 only: REVOKE, MREV and DROP decode illegal under `CAPSTONE_SPATIAL_ONLY`.
- The submodule pins equal P1's.
- Step A's synthesis-only gate on this flow (board/meas-pw 2f6d63a7e6eb): core -7,774 LUT, -3,696 FF post-synth.

**Keys** (`graders/grade_full.py` against P1's seal, plus the rtl lane's bands):
- placed LUT [177,068, 182,068] (P1 - 7,800 ± 2,500); placed FF [70,843, 74,843];
- routed clean; placed congestion ≤ 5; check_timing loops 0;
- WNS against P1's null band [-16.913, -9.433]. Above the band is reported as such; below it goes to the worst-path
  families before any flash.

**Memory.** Step A's gate peaked at 32.02 GB guard RSS, against P1's 27.35, unexplained. This run is under the same 40 GB
ceiling, and the guard's log is read at launch and at the end. A kill at the ceiling is recorded with its phase, and
there is no retry with a higher ceiling.

**Failure branch.** On a placement or routing failure, placed area and congestion only are recorded; no WNS is read and
nothing is flashed. There is no retry and no flow change. The same holds for a guard kill.

**Name.** `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-spatialonly_1f1ed11f9.bit`.
It is sealed and handed to the runtime lane. Its board arm boots the CSCHECK-emulation firmware variant
(`-DCSC_SPATIAL_ONLY`), the rtl lane's stated second variable.

## Reading: routed and sealed; every rtl-lane key passes, and WNS lands ABOVE P1's band at +0.170 ns (apollo, tag spo-1f1ed11f9, 2026-11-20)

Launched 15:13:34Z, sealed 16:56:44Z as
`caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-f2-hpmtag-lrscfix-tagoverlap6c-pagedtbl-spatialonly_1f1ed11f9.bit`.
- sha256 d16c21977962e3d6ad2ee03adeb7eff4b4843b3c21c76a8e58e305201c48140e; the stored copy hashes equal;
- tarball 235c4ce2745b…; PROVENANCE HEAD 1f1ed11f9;
- guard exit 0 at a 32.02 GB run peak, the same as step A's gate (P1's full build 27.39 GB). It is inside the ceiling and
  stays unexplained;
- in-flight loops-only PASS.

| Key | Reading | Band | |
|---|---|---|---|
| placed LUT | 179,661 (88.16 %, P1 -7,707) | [177,068, 182,068] | in |
| placed FF | 72,847 (-3,696) | [70,843, 74,843] | in |
| placed congestion ≤ 5 | 4 (P1 4) | | PASS |
| check_timing loops | 0 | | PASS |
| routed clean | 0 errors, 232,712 / 232,712 | | PASS |
| WNS against P1's null band [-16.913, -9.433] | **+0.170 ns, TNS 0.000, 0 of 214,985 endpoints failing** | above the band | reported as such |

Slices are 50,422 (98.96 %), 378 fewer than P1.

**Timing, read at the routed report.**
- "All user specified timing constraints are met."
- The core clock is `clk_out1_xlnx_clk_gen` at 40.000 ns (25 MHz), the same as P1's, and `corev_apu/` is unchanged
  against P1.
- That clock's own WNS is +0.523 ns, against P1's -13.173; the design's +0.170 is another group's.
- What carries P1's 13 ns is not established by this reading. The worst-path families of the two builds are the next
  question, and it is the rtl lane's.

**The instrument.**
- grade_full.py printed NO DATA: the router printed no "Effective congestion level" block, which it prints only when it
  finds congested clusters. Its routing task and all its phases are in the log.
- The grade above is from a local copy, `grade-board/grade_full_nocongline.py`. It differs in one reading: with no block
  printed, router congestion reads "none printed", but only if the routing task ran.
- On P1 against itself, the copy reads 4/4, as the original does.
- With it, grade_full.py's own key list reads 3/4. Its WNS key FAILs for leaving the band upward, which the
  pre-registration reports and does not count as a failure.

The seal goes to the runtime lane. Its board arm boots the `-DCSC_SPATIAL_ONLY` firmware variant, the stated second
variable.
