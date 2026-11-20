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
