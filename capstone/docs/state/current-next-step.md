# Next step

## 0. CURRENT — 2026-09-10. The reclaim is implemented and measured; two things are waiting on people, not on work.

1. **The `end`-convention re-ruling — the project lead's.** The earlier ruling was superseded the
   same day. It gates the spec amendment and the flash. It does **not** gate the firmware change,
   which is identical under both surviving routes, so that did not wait.
2. **Push monitor commit `0a5c3d9`.** It is LOCAL ONLY. `project-starch/capstone-sbi` returns
   `403, Permission denied` for this credential — tried and recorded 2026-09-10, so this is a
   provable blocker rather than an assumed one. Someone with write access there has to push it, or
   the credential has to gain access.
3. **What is runnable without either.** The R-30/R-31 RTL pair (`1bfff7776`) is built and clean with
   all three pre-registered falsifiers held, and the firmware half is now gated, so a flash is a
   decision with both halves in hand rather than an emergency. After a flash, the reclaim path
   becomes measurable end to end for the first time — `RCLM` climbs instead of reading zero — and
   the `fillcost`/`fillnop` pair should be re-run, because the monitor's loop is one instruction
   shorter per iteration there (the UNINIT cursor advance walks the pointer for free).
4. **A second `fillcost` draw.** The 23.6 cycles/store figure is N=1 on a system with known
   nondeterminism, and the rung is now cheap to re-run inside any boot that is happening anyway.
5. Boot sw52's lost arm — the `instret` counter probe — needs no new work: the classifier defect
   that killed it is fixed and negative-tested, so it rides along on the next boot.

---


## 0. CURRENT — 2026-09-09. Phase B closed on the shipping firmware (boot sw38); follow-ups in `docs/plans/after-phase-b.md`.

The unification (one branch `capstone-bootstrap`, `make TARGET=fpga|qemu`) is done, pushed and
validated on both targets (see `current-state.md` and `docs/plans/monitor-unification.md`).
Phase B (the plan doc's convergence backlog) is done for items 1–9: geometry, M-2 bound (raised to
96 AND bounded, the lead's choice), the pre-carve refusal, fence.i pruning, the null-blk package
and loader, ONE `create_domain`, and the transferred slot becoming a hole on the board too — each
its own commit with its own gate, five board boots (sw33–sw37) all at the oracles with zero fault
tags. The QEMU tier on the last QEMU-visible change (5A; 5B and 2B are byte-identical there) came
back 17/18: one BEEBS case (sglib-hashtable) printed nothing for 20 s before the loader's first
line, in a window that also produced five boot-to-login infra retries under another user's load;
rerun alone, first in a fresh boot, on the rebuilt images it passed 3/3 at its marker — recorded
as an infra flake, not a monitor result. Item 10 (kernel unification) is deferred by
the lead's decision; the +1.05 MB initramfs stays. What remains:

0. **Done since:** the Phase B tail is pushed (2026-09-09); closing boot sw38 9/9 on the final
   firmware; the nightly marks suites that ran under other users' load; Q-06 localised to `sbi.dom`'s
   `query_region` on a CPMP-resident region (owner unchanged); R-26 with the RTL lane
   (`docs/plans/after-phase-b.md`).
1. ~~**Push the Phase B tail**~~ DONE (`/tmp/capstone/push-final.sh`, the lead's credential; every push a
   fast-forward, preconditions checked): monitor 4a12d8b→5b27d01 (three commits), wrapper
   5450a2d→1c48f02 (three), buildroot 8c51969→d3c2402 (three), caplifive-system 20fd22f→77eb5a8
   (one); then `dev`. The live QEMU images are already rebuilt from the committed source
   (fw_jump c1a450ac5d06, sbi.dom f48906bf25a4; generated file byte-identical to 5A's).
2. Small follow-ups, each its own commit: the dead `mem_l`/`mem_r` locals in `split_out_cap`; the
   `gpoff == 0` `create_domain` branch has no board image (unexercised on silicon, stated in the
   plan doc); R-26 (CCSRRW vs younger CAPSTONE_DYN readers) awaits an RTL-lane demonstration;
   Q-06 (null-blk split S-mode init) belongs to the null-blk owner.
3. Before any SLT or M-2 board boot, restage `/tmp/capstone/overlay-attic/*` into the overlay and
   `build/target/test-domains` (the preflight refuses unused overlay files above its budget).

The S-12 material below is FINISHED BUSINESS, retained as the evidence trail; do not act on it.

---

## (superseded) 2026-09-04. S-12 IS CLOSED. The next steps are no longer about S-12.


**Everything below this section, including the one dated 2026-08-29 that calls itself CURRENT, is
the S-12 investigation and is FINISHED BUSINESS. It is retained as the evidence trail; do not act
on it.**

### What happened

S-12 is root-caused, fixed in RTL, synthesised, flashed, and the SQLite domain that trapped now
completes. The registry entry (`ref/ISSUES.md`) and the full mechanism
(`capstone/tests/fpga-repros/S12-wherecode-notcap-operand-vs-memory/S12-explanation.md`) carry the
detail. The resident bitstream is **`caplifive_r25r26r27_66c4e7517.bit`** (RTL `66c4e7517`; carries the R-25, R-26 and R-27 fixes). **CORRECTED 2026-09-10** — this said an older bitstream, and the board scripts that actually ran boots sw48-sw51 all set `FPGA_BITSTREAM=caplifive_r25r26r27_66c4e7517.bit`, which is the record that settles it. Several drivers still carry stale hardcoded defaults, so **set `FPGA_BITSTREAM` explicitly** or the resident-silicon guard hard-stops.

The verification is **consistent with fixed, not proven**: 4 clean draws against a pre-fix arm that
trapped 3 of 4, Fisher p = 0.071. This project has already ruled that bound insufficient
elsewhere.

### The actual next steps, in order

1. **Two more board draws on the S-12 arm.** ~15 minutes. Takes p from 0.071 to 0.0095 and lets
   "fixed" be claimed properly instead of "consistent with fixed". Cheapest open item by a wide
   margin.
2. **The post-`capenter` inertness trace** (RTL lane). The measurement licensing the flashed
   bitstream was taken on `capldc`, where the switcher idles; our workload enters via `capenter`
   and runs its body after a switch. The flash worked WITHOUT that evidence, not because of it.
3. **The control build**: base `80843404c` + tie-off, no fix, no instrumentation (synth lane,
   needs the lead's authorisation in that session). Decides whether the stale debug tree can be
   dropped permanently for a further 1.820 ns — worth more than the fix itself gained.
4. **`plans/instrumentation-cleanup.md`** — deliberately deferred until S-12 closed. That trigger
   has now fired; it is unblocked.
5. **The `ptr-diff-signed.ll` coverage gap** from the c128 merge: D's version rewrote the IR under
   test from i128 to i64 and deleted a case, so its `CHECK-NOT: __divti3` can no longer fail.

### What is NOT a next step any more

The board-instrument work below — the `mtval` positive control, the recorder bitstream, the
operand-mux readers, "instruments first" — was executed and superseded by the 2026-09-03 root
cause. It is history.

---

---

## Older, superseded next-step blocks

Every superseded "next step" layer — the S-12 evidence re-audit, the causal-trigger and
compiler-first framings, the S-07 reflash sequence, the R-18 workaround list and the 2026-08-05
bitstream notes — is preserved verbatim in
**`history/04-09-2026_17-00-00_next-step-superseded-layers.md`**.

Split out on 2026-09-04. Several of those blocks were still headed `## 0. CURRENT`, so a reader
skimming for "CURRENT" found four of them and no way to tell which was live.
