# Next step

## 0. CURRENT — 2026-09-11 (afternoon). The board work is banked; six repositories cannot be pushed from this credential, and that is now the binding constraint.

**Banked on silicon today, four boots.** sw55: a **130 MiB** capability region created, mapped and
round-tripped — 32x the buddy allocator's `MAX_ORDER 10` ceiling. sw56/sw57/sw58: speedtest1 across
seven testsets in a capability domain and natively, the domain's own instruction count measured on
silicon for the first time, and the position question settled. All of it is in
`ref/fpga-silicon-measurements-for-paper.md` §7f–§7k, whose PRECISION block at the end governs how
many digits any of it may be quoted to. **§7j is the measurement of record**; §7k is the first
pairing and where the tick analysis lives.

**Three results worth knowing without reading the sections.** The domain's instruction count is
**bit-identical** across two boots and matches the emulator to 5e-08. **Position is worth 0.016%** —
`main`'s domain arm run twice in one boot, instruction counts identical, so the positional caveat on
every cross-boot comparison is retired. And run-to-run variation is **0.027 pp sd on the same
image**, six times smaller than the first bound, which had been dominated by a 144-byte layout
difference rather than by the machine.

**M-6 is fixed and M-7 is what was behind it.** `revoke_region` now returns `0` revoked / `2` nothing
to revoke / `-1` refused, and the module treats `2` as permission to pop (the lead's ruling). The
QEMU abort is gone. The release path then hits a second defect — after the pop an access finds no
CPMP region, and the recovery path assumes a domain is running when none is. **`pre_mmap_offset` is
still unproven**, now by M-7 rather than M-6.

**The firmware half of R-30/R-31 is committed at last.** All four monitor pointer paths were
committing the pre-reclaim parent while every boot used the working tree. Verified from a build: five
reclaim sites in the generated asm, and the built firmware's monitor source MD5-identical to what the
tree commits.

### Open, in the order they are likely to move

1. **SIX REPOSITORIES REFUSE THIS CREDENTIAL, and the monitor is one of them.** This is the binding
   constraint, not the board and not the work. Verified against the remotes rather than against
   `@{u}` — three repos have no upstream configured and the tracking comparison returns zero for
   them silently:

   | repo | unpushed | note |
   |---|---:|---|
   | `capstone-sbi` / `caplifive-sbi` (the monitor, two copies) | 1 each | carries the M-6 fix AND the R-30/R-31 firmware half |
   | `capstone-opensbi` (the wrapper, two checkouts) | 1 each | the monitor gitlink |
   | `caplifive-system-dev` (two checkouts) | 2 + 1 | `sw/buildroot` pointers |
   | `capstone-academic-spec` | 2 | the `end`-convention amendment; 403 on **read** too, so its branches cannot even be listed |

   `capstone-qemu` was in this list and is **not** — it was a misconfigured upstream, and its two
   commits (the `badaddr` and cause-6 fault-path fixes) are now pushed. Establish access by trying
   the remote, not by inferring from a tracking ref.

2. **Two branches need an allowlist entry**, which is the lead's file and no lane may edit it:
   `speedtest1` (50 commits, ~3,600 lines — the entire apparatus behind §7f and §7i–§7k, on no
   remote) and `shrinkto-size-fix`.

3. **The flash of `1bfff7776`** — authorised, `.bit` still on the synth machine. It now has a
   **batch** owed to it: control, then the bridge arm tying post-flash numbers to pre-flash ones,
   then the M-3/M-4/Q-06 board controls, the 4 MiB allocator-provenance arm, the `fillcost` repeats
   and sw52's lost `instret` arm, and the R-30/R-31 directed tests LAST. Seven separate items each
   waiting on "the first post-flash boot"; one load, not seven.

4. **M-7's mechanism is not established.** `pop_region` clearing its bookkeeping without releasing
   the CPMP entry is a hypothesis from reading code, recorded as such. The offset proof and the whole
   release path sit behind it.

5. **R-32's ruling** (`SPLIT`, `LCC` bound values) is the lead's. The decision note is at
   `/tmp/capstone/lane-notes-2026-09-10/` and **will not survive a reboot** — moving it into the repo
   is on the list.

6. **The paper numbers resume after the silicon work**, by the lead's ordering. The other lane's
   region-arena change is the single piece of work that lifts both `json` and the depth axis; `app`
   is root-caused and stays explained rather than worked around, written up as a porting-cost result.

7. **Hygiene, none of it urgent:** `board-results/` holds only `2026-09-05.tsv`, so sw52–sw58 are
   absent from the TSV corpus the verification rule requires; Q-07's registry entry, its decision doc
   and its in-source comment all contradict git, which shows it committed and pushed;
   `build-sqlite-host.sh`'s provenance line is on the other lane's branch, not on `dev`.

---

## (superseded) 2026-09-10. The reclaim is implemented and measured; two things are waiting on people, not on work.

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


## (superseded) 2026-09-09. Phase B closed on the shipping firmware (boot sw38); follow-ups in `docs/plans/after-phase-b.md`.

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
