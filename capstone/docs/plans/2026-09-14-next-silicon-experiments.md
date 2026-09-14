# Plan: the collaborator's PR queue, closed out — and the next silicon experiments (2026-09-14)

## Context

**Where things stand (2026-09-14, morning).** Every collaborator PR that could be landed is landed
(twenty-seven llvm-capstone merges, three caplifive-buildroot, one capstone-qemu); one is held for a
rebase and two August ones are on no branch. Boot sw73 closed the size-100 pair: `main --size 100`
on silicon at **1.1819 → 1.18**, exactly the pre-registered value (§7p), on top of 1.2195 (size 1)
and 1.1940 (size 20). The instrument work behind it is done: S-15's share readback with both
controls, R-33's rounding log with its positive control, the eight private-struct hosts grown and
pair-proven, the arena gate refusing non-representable sizes, the runner reading a host-reported share
trap as a result. The paper has a proposal on the table (no edit made).

**What the last boots changed for the next experiments.** Two things fell out of the closing work
that move the Sublet measurements: (1) under the #3 module the non-representable 1,419,584-byte Sublet
arena is rounded up (R-33's log fired for it on sw72), and cell 6's image now reports `HEAP 911104`
instead of the 910,008 every Sublet row was taken at — so the **silicon Sublet pair (sw60/sw61) was
measured on a non-representable arena, before the module rounded it, and needs re-basing**; (2) every
silicon SQLite row is lookaside OFF, a configuration SQLite does not ship, and since `df3c1944` the
switch is a choice — **no lookaside-ON silicon pair exists yet**, and the measurements doc's
CONFIGURATION block names that as the gap between our rows and any comparison with a shipped SQLite.

**Owed but never run** (named in the state doc / plan as controls of their own): the entry watchdog's
live positive control (sw64's image `23da3b126a304585` stalls deterministically at share3; the abort
must be shown to fire without a reset), the R-30 denominator pair (`rtpc` at two region sizes —
runnable now that `rtpc` is grown and pair-proven), the size-100 instret image (sw73 was cycles-only),
and R-33's bottom-truncation arm.

**Decisions this plan waits on from the lead (not blocking the experiments):** the paper proposal
(`docs/plans/2026-09-14-paper-evaluation-update-proposal.md`), the M-1 trap-vector default
(ISSUES.md), closing PRs #14/#18 on GitHub (landed by content), and whether llvm #2/#3 (August) are
abandoned.

## 1. The PR table — every collaborator PR, with what landed and what was measured

Order and grouping as the lead listed them; "control" means a run made here before the pass was
believed. Merge hashes are on `dev` (`git log --first-parent`).

| # | group | PR | state | landed as | measured here |
|---|---|---|---|---|---|
| 1 | Toolchain and compiler | llvm **#15** — `toolchain-fresh` no longer switches the staleness check off | **LANDED** | `376486fca526` (2026-09-13) | the early return reproduced on dev before the merge (rc=2 at the drift, staleness never checked); B8: full-target rebuild → FRESH, its `opt`/`llvm-symbolizer` targets were unbuilt until then |
| 1 | Toolchain and compiler | llvm **#14** — a capability parked in `ra` spills as a capability (S-14) | **LANDED** — branch later force-pushed as a pure rebase: **CLOSE, do not re-merge** | `d616ea4e20eb` + follow-up `bc982fe3f45a` (lit test's claim corrected) | rebuilt pre/post: the 160-test image differs by exactly 17 instructions (13 `sd ra`→`stc ra`, 4 `ld ra`→`ldc ra`, all in `__capstone_cap_init`); lit 93/93, authority 32/32, sqlite-silicon PASS, MicroPython PASS=4; claim-auditor SUPPORTED; `capinit-scan` cannot see the spill form (B4) |
| 2 | Domain data budget | llvm **#16** — `domdata-budget` models the module's allocation | **LANDED** | `0d9a5091b48e` + follow-up `47f3ba497638` (stale print) | control on real images (before: order 9 reported where the module allocates order 8, and DOES NOT FIT for the 1.48 MB speedtest1 image that loads); B6: the ceiling-order convention is read from the kernel header, synthetic FORCE_MAX_ORDER=10 → 10 (inclusive) / 9 (exclusive) |
| 2 | Domain data budget | llvm **#17** — models the declaration now that the module consumes it | **LANDED** | `215c481268c6` + follow-up `db7d16c80f88` (dead paths dropped) | control on real images: cell 5 DOES NOT FIT order 11, fix image order 10 |
| 3 | Loader and kernel module | buildroot **#2** — the loader reads the declared requirement (`e618b89`) | **LANDED** (one submodule merge `d04bd83` lands #2 and #3; gitlinks `8c6db339acc6`; buildroot #1 = the CMA-regions merge `c64f09b` of 2026-09-10, whose message carries no number) | plain merge in the submodule | module rebuilt + three QEMU controls (new host runs; OLD host `Unrecognised IOCTL`; over-declaring image refused at order 11) |
| 3 | Loader and kernel module | buildroot **#3** — the module sizes one region from the declaration | **LANDED and PROVEN on silicon** (boot sw71; sw70 VOID = staging mistake, fixed with a same-program gate) | same | Finding A (ioctl-struct skew: every private-struct host grown, nine total) and B (over-declaration under the one-region rule; the stack-declaration knob is diagnostics-only); the `-ENOTTY` and old-size-accept asks are in the hand-off note |
| 4 | PostgreSQL | llvm **#11 → #12 → #13** — Sublet primitives to a shared home; PostgreSQL port | **LANDED** (one merge for the stack) | `36f9ca8767d4` | #12 ships `run-pg-gate.sh` (four checks, generates its workload), #13 `run-pg-sublet-gate.sh` + a level-below unit test; both gates are in run-nightly; the merge records the diff clean under precommit-scan and no numeric gate line — the gate numbers live in the nightly reports, not in a commit; #12 documents files only #13 adds — landed together |
| 5 | MicroPython | llvm **#18** — MicroPython runs in a domain, build declares its need | **LANDED** — branch later force-pushed as a pure rebase: **CLOSE** | `8c773a62f851` | gate `PASS=4 FAIL=0` (the 160-test `PASS=100/FAIL=60` figure in the PR still owes its log) |
| 6 | Simulator | capstone-qemu **#3** — the store-address watch (`diag/store-address-watch`, tip `66bb74699a` 2026-09-13) | **HELD** — forks from `44a7f8ab3d` (2026-08-26), before the diagnostics merge `656cc03489`; carries a duplicate of `b59d116983`; rebase onto `c128-qemu-merge` asked, dropping the two commits already landed in superset form | — | qemu's log names no PR number for #2; the 2026-09-11 diagnostics merge is the landed content the 2026-09-13 plan recorded as #2 |
| new | standalone | llvm **#20**, **#21**, **#28** | **LANDED** | `ba8be847bc51`, `297efdca89ba`, `b0c404321c0f` | #20's selftest passes here (every case behaved); #21 is data only, its "96 lines shorter, refutes H4" recorded as the collaborator's claim; #28's gate exits 0 on the fix and Sublet images (no false positive) and its fire-on case is the nginx port |
| new | nginx | llvm **#29 → #30 → #31 → #32 → #33 → #34 → #35** | **LANDED** in order | `7b3bc8533e42`, `712f4b1c985b`, `0ebd8823692b`, `33d5e9be8b84`, `0b99ef2f4c1b`, `8896fe9bfa64`, `742beec11e76` | `run-nginx-domain.sh` checks 90, failures 0 at the tip (the PR text says 78; #33 adds the twelve); #32's use-after-destroy pair reproduced (plain stop1/2/3 = C10000/C20000/C300A0; Sublet stop 3 = FAULT, untagged, cause 24, at the reported pc 0x166cc); #34 format-only; **#35's replay NOT reproducible here** — it needs a `.ngxt` trace from the paper's recorder, in neither the repo nor the PR; recorded as the collaborator's claim |
| new | MicroPython stack | llvm **#24 → #25 → #26 → #19 → #22 → #23 → #27** (rebased by the collaborator onto dev, 0 replayed commits) | **LANDED** as plain merges | `0049081780b7`, `29e1f4f57862`, `6d4cffa0f7cf`, `a1c802413d69`, `f5d6693fd906`, `7170a00ff23e`, `c015b6df344a` | gate on the tip: default PASS=4; full level 557 rows PASS=551 FAIL=1 FAULT=0 SKIP=5; weakref-on 563 rows PASS=552 FAIL=1 FAULT=5 SKIP=5 (#26's comment: 559 / 549 / 1 / 4 / 5, before #19 widened the selection). **Defect:** the gate's `MPY_GATE_TESTS=all` mode cannot fail (its judge compares to the literal `all`) — in the hand-off note |
| Aug | — | llvm **#2**, **#3** (2026-08-22) | **on no branch, not on dev**; no commit on dev names them (the August landings are the unnumbered i128/capability codegen merge `b25a4cab9582`, lit 47/47, and the numbered `#1` nightly-gates merge `c15f42470ac0`) — abandoned or forgotten; the lead says which | — | — |

Hand-offs to the collaborator (through the lead; `/tmp/capstone/pr-review-notes-for-the-collaborator-2026-09-13.md`,
never committed): the qemu #3 rebase; module `-ENOTTY` + accept the old `DOM_CREATE` size + fall back
with a `pr_warn` above the order ceiling; #14's lit test does not gate the defect; #18's 160-test log;
the MicroPython gate's `all`-mode judge; the nginx replay trace; the share-entry trap's proper fix
(monitor + wrapper, no ABI change).

## 2. Next experiments — ordered by evidence per board-minute; the board runs are short boots now

Every board arm keeps the discipline that made sw71–sw73 clean: control first, one unknown per boot
and last, pre-registered readings in the driver header, results cited by image hash from the
run-scoped log, entry watchdog on, `BUDGET` set per arm. **Distinct SQLite images share one entry VA
(0x10000) and cannot share a boot** (preflight C15), which is what sets the boot count below.

### E1 — QEMU first (minutes, no board): re-base the Sublet cells under the #3 module, and record the passes the board preflight needs

* Run cell ⑥ (`ceeded2533a74bce`, archived) and its control ⑥′ (`5f5045dbe075d458`) through
  `run-speedtest1-measure.sh` (icount, Sublet mode: `--arena 1419584 --tables 1750285`, lookaside
  1200,40) on today's rootfs with the #3 module. Pre-registered: oracle hash; **`HEAP 911104`** (the
  rounded arena, measured 2026-09-14 with the rebuilt rr host), not 910,008; the instruction count
  differs from the archived 690,505,703 / 705,994,514 by the arena geometry only — record the new
  counts as the #3-module rows in §4g, beside the old ones, never in their place.
* Build ⑤ (lookaside 1200,40, memsys5, 2 MiB static heap) with `SPEEDTEST1_STACK=385024` so it loads
  under the one-region rule (the knob is diagnostics-only, proven byte-neutral) — the measure script
  writes the hash-keyed QEMU pass the preflight requires; oracle hash, `HEAP 2097152`, count
  678,572,868 ± jitter (⑤'s archived count; the S-14 fix changed nothing at steady state, as ④ showed).
* Build the **lookaside-ON native baseline** (`SQLITE_LOOKASIDE=1200,40 build-speedtest1-baseline.sh`)
  and run it under icount at size 1: oracle hash, `Successful lookasides:` non-zero (② showed 25,122),
  count near ②'s 535,335,376.

### E2 — Boot A (short): the R-30 pair, the re-based Sublet cell, and the lookaside-ON baseline

Arms: control `k800` → `rtpc` with `rev_transferred_probe.dom` at region size 1 → `rtpc` at region
size 2 (the R-30 denominator pair: the reclaim shortfall at two sizes says whether it scales with the
region or is a constant; both return in seconds; pre-register the `RESULT revxfer retval=` form the
QEMU pair printed) → **⑥ Sublet, size 1** (the only SQLite domain image of this boot; pre-registered:
oracle, `HEAP 911104`, cycles against sw61's 2,797,516,229 — the difference is the geometry term,
written down as such) → `speedtest1_baseline` lookaside-ON at size 1 (native; oracle; cycles against
§7m's lookaside-OFF baseline) → trailing control. Nothing in this boot is expected to stall.

### E3 — Boot B (short): the lookaside-ON domain at size 1, so the "as shipped" pair exists on silicon

Arms: control → **⑤ lookaside-ON domain, size 1** (`SPEEDTEST1_STACK=385024` build from E1) → control.
Pre-registered: oracle; `HEAP 2097152`; `Successful lookasides:` non-zero in the payload; ratio to
Boot A's lookaside-ON baseline ≈ the QEMU ⑤/② ratio 1.2676 scaled by the silicon CPI ratio — write
the band before the boot from §4g and sw63's CPIs. With ⑥ (Boot A) at the rounded arena, **⑥/⑤ on
silicon becomes the re-based configuration cost**, and the state doc's "9.6 % on silicon" row is
replaced, not amended.

### E4 — Boot C (short, by itself): the entry watchdog's live positive control

Control → sw64's image `23da3b126a304585` **last** (deterministic stall at share3, `SHA5` without
`SHA6`). Pre-registered: the watchdog aborts on UART-line liveness within `ENTRY_STALL_S` (≥ 260 s)
**without a board reset**, the run-scoped log carries the abort line, and the driver releases the
board cleanly. A boot of its own because the image cannot share a boot with any other SQLite image
and cannot be relinked (the stall is per-image). Until this runs, every "the watchdog would have
caught it" sentence is unproven.

### E5 — Boot D (~1.5 h, decided by the lead 2026-09-14): the lookaside-ON pair at size 20

`main --size 20` with BOTH arms lookaside ON — the "as shipped" SQLite ratio at a size where the pool
matters, beside sw68's 1.1940 (lookaside OFF, same size, same image family). Build: the domain is the
fix image's recipe (`cc55013c2106`'s flags — `SQLITE_FULL=on SQLITE_FLOAT=on SPEEDTEST1_REGION_ARENA=1`,
128 MiB arena, `mtvec = 0`) plus `DOMAIN_EXTRA_DEFS=-DSQLITE_DEFAULT_LOOKASIDE=1200,40`; the baseline
is sw64's baseline recipe plus `SQLITE_LOOKASIDE=1200,40`. QEMU first (icount, size 1 and size 20;
oracle `3807866 2738af78` at 20 — lookaside does not change the verification hash; `Successful
lookasides:` non-zero on both arms; the pass record written). Arms: control → baseline ON, size 20 →
control → **domain ON, size 20 LAST**; `BUDGET` from the QEMU counts × sw68's CPIs. Pre-registered
before the boot: both hashes at the oracle, `DROPPED 0`, the lookaside line on both arms, ratio band
from the QEMU ON/ON instruction ratio × sw68's CPI ratio (0.952), two decimals; the reading is written
beside sw68's 1.1940, and the §7 CONFIGURATION block's "no lookaside-ON row" sentence is closed by it.
Boot order: A → B → C → D (the measurement boots first, the stalling image before the long pair).

Not scheduled (listed so they stay visible, the lead's call later): the size-100 instret image
(~3 h; board CPI beside the QEMU count, the ratio is already carried by cycles); seven testsets at
size 20 (~9 h, seven boots); R-33's bottom-truncation arm (its shape is not written yet).

### E6 — conditional on the lead's answers (no board)

* Paper: if any part of the proposal is approved, the edit is small and local to
  `parts/evaluation.tex`; never push `capstone/paper`; the parent's pointer stays unbumped.
* M-1: if measurement images are to carry the trap vector, that image is a new binary and needs its
  own bridge pair (size 1 on QEMU + one boot) before any row uses it.
* Collaborator queue: land capstone-qemu #3 when the rebase arrives (plain merge, storewatch runs on
  QEMU as its control); close #14/#18; record the llvm #2/#3 decision.

## Verification

* E1: each run hashes to the off/off oracle `112006 38bb59fd`; ⑥/⑥′ report `HEAP 911104`; ⑤ reports
  `HEAP 2097152` and the lookaside line; the baseline reports lookasides; every count is icount
  (`-icount shift=0,sleep=off`, the measure script's default) and the pass record exists before any
  image is staged.
* Every board boot: control `retval=4` first (VOID otherwise), readings from the run-scoped log after
  this run's own `load_image`, hashes matched to the staged files, the pre-registered band written in
  the driver header before launch; the same-program gate and the QEMU pair before any host is staged
  (the rr-host rebuild that passed the string gate and failed the region check is why).
* E2/E3: ⑥ and ⑤ on silicon at the oracle; the new ⑥/⑤ carries the geometry caveat explicitly (arena
  1,421,312 rounded vs 2 MiB static heap) and is labelled "configuration", never "discipline".
* E4: the abort line appears, no `SQ: G/enter`, the board comes back for a trailing control without a
  power cycle; if it resets, the watchdog is not yet the instrument it is claimed to be — record that.
* E5 (Boot D): the QEMU pre-run at size 20 hashes to `3807866 2738af78` on both arms with a non-zero
  lookaside line; on the board both hashes match, `DROPPED 0`, the lookaside line is in both payloads,
  the ratio lands in the pre-registered band; the row is labelled lookaside ON and compared only with
  sw68's OFF row of the same size and image family.

## Documents to update (the last step of each item, committed with it)

* `docs/ref/fpga-silicon-measurements-for-paper.md`: §4g gains the #3-module rows for ⑥/⑥′ (E1) and
  the lookaside-ON rows (E1/E3); a §7q for Boots A/B/C; the CONFIGURATION block's "no lookaside-ON row"
  sentence closed when E3 lands.
* `docs/ref/ISSUES.md`: R-30 gains the denominator pair; R-33 the rounded-arena geometry note; M-1 the
  watchdog control's outcome.
* `docs/state/current-next-step.md` / `current-state.md`: the PR table above in its final form; the
  Sublet silicon row re-based; the open list reduced to E5/E6.
* This plan lands in `docs/plans/` as the dated successor of `2026-09-13-pr-queue-and-sqlite-experiments.md`.
