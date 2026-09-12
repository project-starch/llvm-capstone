# Current Capstone state

Minimal snapshot. Read first in every session.

## 2026-09-13 — CURRENT

> The bridge is discharged. §7f–§7k now carry forward to the flashed bitstream; the 2026-09-12 block
> below stands for R-30/R-31/R-33.

* **THE BRIDGE HOLDS (boot sw63).** §7k's images re-run **unchanged** on
  `caplifive_r30r31_1bfff7776` — verified byte-identical after the bake, from both `overlay/` and
  `build/target/`, which mattered because the overlay was holding the lookaside matrix builds under
  those very names. Every §7k-comparable pair agrees to within **0.07 pp** against the 0.171 pp
  cross-boot band, and `main`, the pair the protocol names, to **0.05 pp** (1.2195 vs 1.220).
  Controls at both ends, **7/7 pairs agree on their verification hash**, `DROPPED 0` throughout.
  **§7f–§7k are carried forward.** Details in §7m. This is the arm sw59 was wrongly reported as.
* **R-33's fix was inert on that boot, and it was proven rather than assumed** — zero
  `not representable` lines. Every constant on the path is a power of two. It would **not** have
  been inert for a `--pool`-derived arena, where both the arena and tables round.
* **The size-20/100 artifacts were rescued from `/tmp`** into
  `~/capstone-artifacts/speedtest1-size100/` with a `SHA256SUMS` and a README. They were
  single-copy, including the only record of their hashes. One set covers sizes 1, 20 and 100 — the
  size is argv and the arena lives only in `HOST_EXTRA_DEFS`, so no rebuild is needed for depth.
* **The `speedtest1` branch is merged into `dev`** (`5e5d9cb42318`). It was on no remote and not on
  the push allowlist. That lands §7l **and `arena-mismatch-gate.py` with its runner wiring** — until
  now a campaign from the main checkout ran with no arena gate at all, and the mismatch it catches
  silently corrupts a ratio instead of crashing a run.

## 2026-09-12 — superseded above for the bridge; current for R-30/R-31/R-33

> Three boots on the R-30/R-31 bitstream. The 2026-09-10 block below is still accurate for the
> firmware half; everything it says about R-30/R-31 being unverified on silicon is now superseded
> by this block. Measurements: `ref/fpga-silicon-measurements-for-paper.md` §4g.1–§4g.5.

* **The bitstream is flashed and verified BY CONTENT.** `caplifive_r30r31_1bfff7776.bit`,
  `nv_bitstream_sha256 = 406e12bf…3b30` read back from a fresh `/api/state` after the mandatory
  power-cycle. Control `k800` = 4 in **all three** boots (sw59/sw60/sw61), `instret` 1089 in every
  one, cycles 4521/4517/4517 — so the flash did not move timing. `known-good-controls.md` is
  refreshed against it; only the `k800` row, the others still carry older bitstreams.
* **R-31 is FIXED ON SILICON** (boot sw60), through the monitor's real share/revoke path rather
  than a fabricated capability: `SHA2:00000003` = cap_type UNINIT where the previous bitstream
  returned LINEAR, and `RCPR` did not fire, so the cursor is at base too. Both halves of the
  contract hold.
* **R-30's headline is SUPERSEDED, and this was over-stated once before being narrowed.** The
  one-byte precondition is fixed — boot sw61 performs **5,334 successful INITs**, every counter
  bit-identical to QEMU. sw60's `RCSH:000006C0` is a *separate* large-region effect one step
  earlier, where INIT refusing is correct. Two of the four candidate accounts are dead: the
  monitor's own arithmetic admits a shortfall of at most 15 bytes (which also kills
  allocator-rounding), and the RTL kills "end moved during the fill". **One account survives: 108
  stores did not advance the cursor.** The discriminator is a pair of boots at two different large
  region sizes — constant 1,728 = a fixed tail effect, scaling = a proportional store-failure rate.
  Not yet run.
* **The silicon allocator matrix is complete.** ABI cost **~1.21**, with the two allocators
  **indistinguishable at this precision** (④/① 1.2124, ⑤/② 1.2107 — 0.17 pp against a 0.171 pp
  cross-boot band, so not resolvable either way; the deterministic QEMU pair *is* resolvable and
  shows lookaside costing marginally less, so "barely depends", not "independent"). The Sublet
  **CONFIGURATION** costs **1.0964 on silicon against 1.0176 on QEMU** — 5.5×, consistent with an
  O(bytes) reclaim. **Not "the discipline":** both pairs carry the heap-geometry mismatch for which
  the QEMU figure was already retracted as a discipline cost, so the 5.5× is suggestive of the
  mechanism rather than a measurement of it. (Both corrected 2026-09-12 after a bench-lane audit.) Two caveats travel with these rows: the ⑥/⑤ comparison
  carries a heap-geometry term (910,008 vs 2,097,152, not equalisable), and the lookaside-ON rows
  must not be blended with the lookaside-OFF §7 corpus.
* **R-33 is a SOUNDNESS issue, ISA-LEVEL, and its cause is the ALLOCATOR.** Demonstrated 2026-09-12 to reach ordinary LINEAR capabilities through `CINCOFFSET` — plain pointer arithmetic — by a matched RTL-sim pair where the representable control does not move and the non-representable arm widens by exactly the predicted amount. The
  rounded `end` is the authority bound — `STC` checks `rs1_up > metadata.end - 16` against the
  decompressed value — so a non-representable region grants writes past itself, and `CINCOFFSET`
  reaches ordinary linear capabilities by the same route. A lossy compressed-bounds format is
  standard and is exact for representable objects; nothing here enforces representability, and
  `create_region(N)` passes `N` through unchanged. **Contained by the kernel's `PAGE_ALIGN` below
  4 MiB and not contained at or above it** — no region used so far escapes (sw60's sits exactly on
  the boundary), but a 130 MiB-class region has a 262,144-byte granule. The over-permissive store is
  **DEMONSTRATED in RTL simulation 2026-09-12** — a representable control's store at its true end is refused OUT_OF_BOUNDS while a non-representable arm's identical store retires without fault (`r33-store-past-end.S`, trap_mask 0x1 as pre-registered). Not yet shown **on silicon**, and the bottom-truncation half is still unexercised.

* **R-30's residual is SOLVED and re-filed as R-33 (boot sw62).** The 1,728 bytes were never a
  failed fill. A capability's bounds are re-encoded once its cursor leaves `base`, and `end` then
  reads high by up to one granule — `compress_bounds` uses an exact form only while the cursor sits
  at the low bound (`ariane_pkg.sv:787`) and otherwise rounds the top up to `2^(E+3)`
  (`:827-828`); `STC` is a DYN op (`decoder.sv:1309`) whose `rs1` is re-compressed on writeback
  (`ex_stage.sv:1188`). sw62's granule-ALIGNED arenas reclaimed clean, while the unaligned arm
  halted with `RCSH = 448` — the compression figure, pre-registered before the boot against 432 for
  a store-failure rate — with `RCCU` showing the cursor reached the true end, so **no store failed**.
  Two accounts were retracted on the way there, both this lane's and the RTL lane's, and both had
  read the fat struct without the function that compresses it.

* **The resident firmware is THREE monitor commits behind and none of them has booted** —
  `75d96d2` (define `CAP_TYPE_UNINIT`), `921f598` (`RCEN`/`RCCU` reclaim instrument), `d1bd7e4`
  (early-clobber on both `C_RECLAIM` outputs). All boots ran `2c49c41`. `board-b59/b60/b61.sh`
  gate on that hash and will now FAIL — correctly; update it deliberately, do not delete the gate.
  The submodule pointer chain is deliberately unbumped, which matters only for a fresh clone.
* **`precommit-scan.sh` had a silent-pass path** — a `--range` git could not resolve contributed
  nothing and printed CLEAN. Fixed (unresolvable *or* empty range now blocks), negative-tested four
  ways. Nothing had escaped it.

## 2026-09-10 — superseded above for R-30/R-31; **2026-09-11 is NOT in this file**

> **Read `state/current-next-step.md` first for 2026-09-11.** Four boots landed that day and none of
> them is described here: sw55 (a **130 MiB** capability region on silicon), sw56/sw57/sw58
> (speedtest1 across seven testsets, the domain's instruction count measured on silicon, and the
> position question settled). The measurements are `ref/fpga-silicon-measurements-for-paper.md`
> §7f–§7k. Also that day: the CMA board half, M-6 fixed and M-7 filed, the R-30/R-31 **firmware
> half committed at last** (all four monitor pointer paths had been committing its pre-reclaim
> parent), and the discovery that **six repositories refuse this credential** — including both
> copies of the monitor and the academic spec, which returns 403 on read as well as write.


* **The reclaim (R-30/R-31 firmware half) is implemented, gated and measured.** The lead ruled
  *fill, then initialise*. Monitor commit `0a5c3d9` adds a `C_RECLAIM` asm loop at **five** sites
  (the annotation branches share a hoisted one, so REV_DEFAULT/BORROWED/SHARED/TRANSFERRED are all
  covered), with the loop bound taken from `cap_end - cap_base` and **not** the cursor, so a
  cursor-at-end arrival faults instead of running zero iterations. Emulator gates green: host-call
  12/12, linear/uninit corpus 7/7, smoke, nullblk 3/3. ~~**`0a5c3d9` is LOCAL ONLY**~~ **CORRECTED
  2026-09-11: it is PUSHED.** `git ls-remote` — the remote itself rather than a cached
  remote-tracking ref — shows `capstone-sbi refs/heads/capstone-bootstrap` at exactly
  `0a5c3d9a3413`. The 403 recorded on 2026-09-10 was real then and was carried forward for a day
  without being re-tried; nobody needs to push this.
* **Boot sw52 — speedtest1 on capability silicon vs native, plus the fill-cost pair.** Control
  `k800` = 4, zero fault tags, 10 of 11 arms. Capability/native **cycle** ratios 1.335 / 1.181 /
  1.214 for parsenumber / orm / main, with **identical verification hashes** on every pair; the cycle
  ratio is BELOW the instruction ratio in all three because capability CPI is *lower* than native's.
  Full numbers: `docs/ref/fpga-silicon-measurements-for-paper.md` §7f.
* **Boot sw53 -- the fill-cost pair with an error bar, and both of §7e's open questions closed.**
  9/9 arms, `k800` = 4, zero fault tags. **24.1 cycles per 16-byte capability store at n=3**
  (within-boot spread 0.2-0.5 %; sw52's single draw of the earlier rung gave 23.6). A 4 KiB reclaim
  is **+3.6 % instructions and 7.8-9.3 % of cycles** at speedtest1's measured CPI, 4.6-26.2 % across
  the full 1.13-6.44 spread -- CPI-sensitive, not "in the lower half of the bracket". An adversarial
  audit had already moved that figure from 5.5-6.6 %: the first version counted the fill loop's
  INSTRUCTIONS but only the stores' CYCLES, and the monitor pays for the loop too.
  - **A warm region is NOT materially cheaper** (`fillwarm`: second pass >= 87 % of the first), so
    the cold-buffer "this is a ceiling" caveat is deleted and the figure applies to the monitor's
    real case.
  - **The cost is per STORE and is ~96 % not capability-specific** (`fillsd`: a plain 8-byte store on
    the same 16-byte walk costs 23.1 against `stc`'s 24.1, writing half the bytes). It is the
    write-through drain, not the tag.
  - **R-3 does not bite the ladder path**: every rung repeated at its own entry VA returned.
  Full numbers: `docs/ref/fpga-silicon-measurements-for-paper.md` §7e, §7f.
* **`RCLM:00000000` on every share (9 of 9).** Zero reclaims, as it must be where the guard cannot
  fire — and the counter's reporting path is now proven readable on silicon, which is the one
  property of it that could not be tested after the flash.
* ~~**Still open and the lead's:** the `end`-convention re-ruling.~~ **RULED 2026-09-10 (evening) —
  adopt the resolution**; `plans/DECISIONS-WAITING-2026-09-10.md:50`. This line said "still open" for
  a day after the ruling landed, which is the same shape as the stale 403 withdrawn on 2026-09-11: a
  blocker asserts a fact about today, so re-verify before repeating it. The spec amendment is
  committed in `capstone-academic-spec` (`21a01f0`, `eadaf87`) and **cannot be pushed** — that remote
  returns 403 for this credential on read as well as write, so its branches cannot even be listed.
* Three tools repaired and negative-tested the same day: `preflight-board-run.sh` (parsed only one of
  the three stage forms, and could not require a host binary at all), the sw52 result parser (needed
  the testset, cycles and hash on ONE line when they are on three), and the driver's staged-marker
  guard (did not know the `BASELINE-PROBE` form, and cost sw52 its last arm).

## 2026-09-09 — SUPERSEDED BY THE ABOVE, still accurate for what it covers

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
