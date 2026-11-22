# Spatial-only step C (gate, then full build) and the B5a demand-only full build (board lane copies of the pre-registrations)

Status: pre-registered 2026-11-21, before either launch. The keys are the rtl lane's, copied unchanged.

## Spatial-only step C: capstone-ariane spatial-only-c 321af31a3 (tag spc-321af31a3)

Pre-registration: land/rtl-spatial-c ed7565cfd5c0, `capstone/docs/plans/2026-11-21-spatial-only-core-c.md`. That file
carries the rtl lane's retraction of step B's census: REVOKE and MREV are in the monitor's `_create_domain`, and step B's
first boot wedged there.

**Tree, read at source.**
- Against step B (1f1ed11f9), `core/decoder.sv` only (+7/-7): REVOKE and MREV are legal again and go to the stub, and
  DROP stays illegal. The commit 321af31a3 itself adds tests only.

**Gate** (`twc/spc/post-spc.sh`, graded by `twc/spc/grade-spc.py` against step A's own gate, sp-cf7a00ce6-gate-out):
- mapping within 10 % of step A's 4,914 MB, inside the 40 GB guard;
- 0 loops, each control firing;
- core LUT within ±2,500 and core FF within ±2,000 of step A's gate.

The grader was dry-run with P1's gate as the trial, and it FAILs LUT and FF as it must.

**Full build on a gate PASS**, graded against step B's routed seal (1f1ed11f9: LUT 179,661, FF 72,847, WNS +0.170):
- placed LUT 179,661 ± 2,500 and FF 72,847 ± 2,000;
- routed, with the routing-failure branch; congestion ≤ 5; loops 0;
- WNS in step B's null band [-3.57, +3.91].

Name: `caplifive_…-pagedtbl-spatialonly-c_321af31a3.bit`, with P1's prefix. The seal goes to the runtime lane (Addendum
211, chain spoc-b1 on board 1).

## B5a demand-only: capstone-ariane b5a-demand-only 2ca17ad2a (tag b5adem-2ca17ad2a), full build

The pre-registration is in the commit message, pushed before any gate.

**Tree, read at source.**
- Parent 7180a83b6 (B5a + R-76).
- `core/ex_stage.sv` only (+4/-1): RC_RWAIT goes to RC_IDLE, so the background fill states are unreachable.
- Lint is identical to 7180a83b6's; its sweep is identical; the two-sided record-cache arms change exactly the fills.

**Keys** (`grade_full.py` against 7180a83b6's own seal):
- mapping in 7180a83b6's class;
- placed LUT 189,920 -200..+500;
- loops 0; routed, with the routing-failure branch; congestion ≤ 5;
- WNS reported against 7180a83b6's.

Name: `caplifive_…-pagedtbl-reccache-r76-demandonly_2ca17ad2a.bit`. Its board pair, against 7180a83b6, is the runtime
lane's.

## Reading: the step C gate PASSES 4 of 4; its full build launched (2026-11-22)

`<mnt-hdd>/synth/spc-321af31a3-gate-out/grade-spc.txt`, against step A's gate:
- TM peak 4,925.0 MB (step A 4,914.4, +10.6);
- no loops by any probe, each control firing;
- core LUT 154,652 (+23) and core FF 50,398 (-1);
- guard peak 32.01 GB.

The full build launched 04:46:02Z (tag spc-321af31a3).

## Reading: the demand-only full build routes and passes every key (apollo, tag b5adem-2ca17ad2a, 2026-11-22)

Sealed 05:28:41Z as `caplifive_…-pagedtbl-reccache-r76-demandonly_2ca17ad2a.bit`:
- sha256 b716f9cd913351907a8213dd3fbbea5c8ccecbd0a09e25637f0e8f0e310ebd6b; the stored copy hashes equal;
- tarball 8a1512963051…; PROVENANCE HEAD 2ca17ad2a;
- guard exit 0 at a 26.15 GB run peak; the collector waited for the headroom floor, then peaked at 13.18 GB.

| Key (against 7180a83b6's own seal) | Reading | |
|---|---|---|
| mapping in class | TM peak 6,856.6 MB vs 6,867.5; growth +8.0 on both (from `synth_1.runme.log`: the full tree keeps no `.vds`) | in |
| placed LUT 189,920 -200..+500 | 190,172 (+252) | in |
| loops | 0 | PASS |
| routed | 0 errors, 245,812 / 245,812 | PASS |
| congestion ≤ 5 | placer 3 | PASS |
| WNS (reported) | -12.349 (7180a83b6 -13.615, Δ +1.266) | |

The seal goes to the runtime lane, for the board pair against 7180a83b6.
