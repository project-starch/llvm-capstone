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
