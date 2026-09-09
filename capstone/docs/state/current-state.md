# Current Capstone state

Minimal snapshot. Read first in every session.

## 2026-09-09 — CURRENT

* **The R-25/26/27 bitstream is on the board** (`caplifive_r25r26r27_66c4e7517.bit`, flashed persistently
  2026-09-09; `caplifive_s12fix_5097eb166.bit` stays in the console store as the restore path). Boots sw44
  (closing set 9/9), sw45 (acceptance 7/7, R-25 PROBE wedges as predicted), sw46/sw47 (firmware variants D/E
  clean — the R-26 board arms, no-regression only), sw48 (the s06agg isolation). R-25/R-26/R-27 archived.
  The monitor drops its four R-26 `fence.i` (monitor `1a39e37`, wrapper `f17110a`, buildroot `fe31893`,
  caplifive-system `884b716`; nested pushes need the lead's credential).
* **RETRACTION: S-06's struct-assignment half was never fixed — now `R-29`.** `s06agg` reads 66 on the
  new bitstream in two firmwares (sw46, sw48); the RTL lane's directed test fails on `5097eb166`,
  `ef5a8eaf2` and `66c4e7517` alike when the plain `sd` of the high half is adjacent to the 128-bit `ldc`
  (write-buffer forwarding, S-07/S-10 family). The S-06 folder's cited acceptance (boot B1 "15") was a
  different program. Memcpy half stands. `W-12` stays. Not a regression of the new bitstream.
* Driver classifier: `created`/`entered` now also read the monitor's `DBAS`/`ENT1` tags, so a wedged
  `rtpc`/`lpc` probe is no longer reported as "domain never created" (sw45/sw47 r25dup).


* **The monitor stack is unified onto ONE branch, `capstone-bootstrap`, in every nested repo**
  (caplifive-buildroot `b7fc740`, opensbi `3de3342`, monitor/sbi.dom `3da7ebe`, caplifive-system
  `fec33fa`; parent `dev` `1a1a3dff5778`, all pushed). One source builds both targets:
  `make TARGET=fpga|qemu`, output in `build-<target>/`, `build` a per-checkout symlink,
  per-target code under `CAPSTONE_TARGET_FPGA`/`CAPSTONE_TARGET_QEMU`. `CAPSTONE_CC_PATH` is
  required. The old two-branch split (board vs QEMU, drifted for six weeks and forced Q-03/Q-05 to
  be ported twice) is gone; the `-board`/`-qemu`/`-unified`/`-dts-65536` names are frozen pre-merge
  tips (ancestors of `capstone-bootstrap`, also `pre-unify/2026-09-08/*` tags). Design and record:
  `docs/plans/monitor-unification.md`; layout: `docs/ref/REPO-MAP.md`.
* **Validated on both targets.** FPGA `.c.S` pair, `fw_jump` and `fw_payload .text` byte-identical
  to boot sw31; **board boot sw32 8/8** (control first, six BEEBS rungs, SLT `select1` identical to
  native, zero fault tags), with the SQLite host program rebuilt against the merged loader library
  so that library has now run on silicon; **QEMU nightly tier 18/18**. sbi.dom now builds from the
  one monitor source (Phase B item 7 done).
* **One nightly regression, surfaced and fixed the same day.** `linear-uninit-corpus` /
  `linear_drop_sibling_ok` failed; bisected to the Q-05 monitor commit (a pre-existing stale
  host-observer read the unification's first nightly exposed, NOT a merge defect), fixed in the
  corpus controller to read back through the domain's alias. Auditor-confirmed; a non-blocking
  monitor-robustness note recorded (ISSUES.md Q-05, 2026-09-08).
* **Phase B collapsed the per-target behaviour (2026-09-08, evening and night).** Geometry on QEMU
  (item 3), M-2 bounded at 96 (item 9, refusal seen on silicon), the pre-carve refusal on FPGA
  (item 1), the rounding and the diagnostic store (6a/6b), eleven of fifteen `fence.i` gone (item
  8, three variants booted), the null-blk package and the relocatable S-mode loader (item 4), ONE
  `create_domain` (item 5, nine differences gone) and the transferred slot a hole on the board too
  (item 2: boots sw36/sw37 ran the first transfer-annotated share on silicon, one HOLE line).
  Board boots sw33–sw37 all at the oracles, zero fault tags; QEMU tier 18/18 through item 3,
  17/18 on item 5A (one BEEBS case silent before the loader's first line amid five boot-login
  infra flakes; 3/3 rerun alone, first in a fresh boot). Left: kernel
  unification (item 10, deferred by the lead), the dead `mem_l`/`mem_r` locals, and the
  `gpoff == 0` loader branch that no board image reaches. Monitor 5b27d01 / buildroot d3c2402.
  **Closed on the shipping firmware 2026-09-09: boot sw38, 9/9** (control, six rungs, SLT select1,
  the transfer probe; one HOLE line; zero fault tags). Next: `docs/plans/after-phase-b.md`.

## 2026-09-05

* **SQLite passes its logic tests on silicon at `-O1`** — the first validation above `-O0`.
  `select1` 1031 records / 1000 queries / 0 failures and `q_two` (the S-12 trigger) both completed
  in a capability domain on `caplifive_s12fix_5097eb166.bit` with the cycle-2 compiler, valid
  control first. Full sweep: **8 boots, 8 valid controls, 19 rung readings, every rung at its
  oracle** (`tests/board-results/2026-09-05.tsv`, compiler lane's branch). RV8 `-O2`, CoreMark
  `-O2` with sibling calls, BEEBS `-O2`, two csmith rungs — all at oracle. C-28's tail-call fix runs
  on silicon, so `-fno-optimize-sibling-calls` can be retired.
* **S-12: 6 of 6 post-fix draws clean, p = 0.033** — see ISSUES.md; the two new draws are `-O1`
  and therefore weaker, so this is strong evidence and still not "proven".
* **The gp-captable miscompute (OPEN since 2026-07-23) does not reproduce** — `rc_p1` = 2080 at its
  oracle. Probable cause **R-20, fixed in hardware by `f623c48a1`**, whose signature is exactly
  that bug's. The blocks it carried (silicon-compatibility claim, branch merge, app-level silicon
  perf) are no longer supported by a live failure.
* **R-20 is FIXED in the resident bitstream** — an alert claiming otherwise was filed and
  **retracted** the same day: the fix is a cherry-pick under a different SHA, and both lanes had
  tested ancestry by hash. Presence-by-content is the check; see ISSUES.md.
* **S-13 does not reproduce at `-O1`**, but bitstream and compiler both changed, so it attributes to
  neither yet.
* **2026-09-07:** Q-03 ported to the BOARD firmware (`fw_payload 44c88d9ebeb1`, audited; boot sw30 7/7 in one
  boot, no exact fit occurred so the hole path is unexercised on silicon and self-reporting); Q-05 fixed in the
  stand-in (the probe observes through the domain; both copies make the transferred slot a hole).
* Q-02 (QEMU build break) closed end to end; Q-03 (position-dependent wedge, reproducible off-board),
  R-25 (INIT linearity break), C-41 (compiler `return` encoding), I-01..I-03 filed and verified.

## 2026-09-04 — superseded by the section above

* **Bitstream: `caplifive_s12fix_5097eb166.bit`** (sha256 `7a97ccd0…62999b0`) — the S-12 fix,
  synthesised and flashed 2026-09-04. It IMPROVED timing over its predecessor: WNS −16.400 →
  −15.311, 987 fewer failing endpoints. Every silicon number taken before it should name the
  bitstream it was taken on.
* **S-12: ROOT-CAUSED, FIXED IN RTL, FLASHED — "consistent with fixed", NOT proven.** A capability
  store's scoreboard rd is aliased to its own store-data register; when it stalls on a full store
  buffer the commit stage holds `we_gpr` while withholding `commit_ack`, the WAW guard clears on
  that write, and forwarding hands the consumer `create_cnull()`. The write happens; the
  RETIREMENT does not. Fix = require `commit_ack_i` in both WAW-clearing clauses, four lines.
  Post-fix the SQLite domain completes 4 draws of 4 against a pre-fix arm that trapped 3 of 4 —
  Fisher p = 0.071. **Two more draws would settle it; until then do not write "fixed" unqualified.**
  Full mechanism: `capstone/tests/fpga-repros/S12-wherecode-notcap-operand-vs-memory/S12-explanation.md`.
* **QEMU is REPAIRED and rebuilt (2026-09-04).** The c128 merge had left `capstone-qemu` unable to
  compile, and because nothing rebuilt it, every QEMU verdict for a day came from a binary dated
  2026-08-27. Three defects fixed in `f5972c364f`; smoke passes and the SLT negative control
  passes, so the comparator is proven able to fire. See `ref/ISSUES.md` Q-02.
  **Two things still do NOT follow from that fix.** The "SLT corpus matches native 15/15" figure
  has no committed harness — it was run ad hoc, so a rebuild does not re-establish it; treat it as
  withdrawn until a re-runnable harness exists. And the nightly still cannot catch a
  non-compiling QEMU. SILICON results were never affected: they came from the board.
* **SQLite CORRECTNESS on silicon: the SQLLogicTest corpus (2026-09-05 → 09-07).** Seven files,
  one boot each, control first, fresh toolchain, every result compared with the native baseline
  from the run's own transcript: negative control and `aggfunc` reproduce their known failures
  exactly (the comparator fires on silicon); `select1/2/3/4/5` identical to native with zero
  divergences — **the whole corpus, 10,807 records, 8,746 checked queries, on silicon.** Rows sw23–sw29 and B8 in
  `tests/board-results/2026-09-05.tsv`; write-up in `ref/fpga-silicon-measurements-for-paper.md` §7c.
* **(Superseded by the line above, kept for the caveat it carries.) SQLite RUNS ON SILICON — that is a LIVENESS result, not a correctness one.** The `slt/`
  corpus executes end-to-end in a capability domain; `s12stress` completes 120/120 prepares and
  15/15 of the corpus matches native under QEMU on the current compiler.
  **Read what that measures.** These files are S-12 *wedge probes*, and they say so in their own
  first lines — `p8_trivial.test`: *"WEDGE PROBE, not a correctness test: expected values are
  dummy, the signal is RETURNED vs WEDGED."* Every table in `s12stress` is deliberately EMPTY,
  because S-12 fires at PREPARE time with no rows processed. So "matches native" is a strong claim
  about **completing without wedging** and a nearly vacuous one about **computing the right
  answer** — the queries mostly return nothing on both sides. Establishing SQLite *correctness* on
  silicon would need a different corpus with populated tables and real expected values, and that
  has not been run. Do not let this line become the citation for a correctness claim.
* **C-19: RESOLVED.** Reading a capability's address now uses a plain move, never `lcc rd, rs, 2`,
  which is not total and traps on an untagged (NULL) operand.
* **The c128 capability value type is MERGED** (external collaborator's branch, 2026-09-04).
  `MVT::c128` replaces i128 as the carrier. Merging it silently reverted C-19 and three header
  declarations; all repaired — see the merge commit. One known coverage gap remains in
  `ptr-diff-signed.ll`.
* **S-06, S-07, S-08: fixed and verified on silicon** (see the 2026-08-16 section below).
* **The debug instrumentation is STALE and expensive.** Every mux reading across the S-12 campaign
  was weak, void or faulted — its own decoder says "UNKNOWN SEMANTICS for this bitstream" — while
  costing 1.820 ns, more than the S-12 fix gained. Every verdict came from software instead.
  `plans/instrumentation-cleanup.md` is now unblocked.

**Next steps are in `state/current-next-step.md` §0.** Sections below this one are retained as the
historical trail; the newest of them is dated 2026-08-16 and predates all of the above.

---

---

## Everything older

The historical trail — the append-only layers from 2026-08-16 back to June, including the S-06 /
S-07 / S-08 bring-up, the R-18/R-19 handovers, the UART retirement and the original overhead
tables — is preserved verbatim in
**`history/04-09-2026_17-00-00_current-state-historical-trail.md`**.

It was split out on 2026-09-04 because this file is the first thing every session reads, and 97%
of it described states that two RTL fixes and a reflash had already invalidated. Nothing was
deleted. When this file and `ref/ISSUES.md` disagree about a defect, **ISSUES.md wins** — it is
the registry; this is a snapshot.
