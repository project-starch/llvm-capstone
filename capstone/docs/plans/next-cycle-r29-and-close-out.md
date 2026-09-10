# Plan 2026-09-10: R-29, the close-out, and five corrections the record needed

## Context

The R-25/26/27 bitstream is flashed and its cycle is closed: R-25 fixed on silicon, R-26/R-27 shipped
with clean no-regression arms, all three archived, the monitor's four `fence.i` dropped, and every
nested repo pushed (`push-final.sh` run by the lead, all four remotes verified in sync at
`1a39e37` / `f17110a` / `fe31893` / `884b716`; parent `dev` at `2203a22f4757`).

Two things then landed that this plan exists to absorb. **R-29's mechanism was audited down to
plausible-not-proven**, so the fix track now needs a separation step before a candidate. And a
verification pass over the seven entries I had proposed closing found that **five of them cannot be
closed on the record as it stands** — one of them because its "secondary defect" is alive and is
R-29's own shape. The bucket I proposed as routine bookkeeping was the least reliable part of the
last plan; correcting it is the first deliverable here.

The lead has decided: split R-29 into its own repro folder now; fix R-24 in the next bitstream; and
settle the four standing items, best-first.

---

## 1. Corrections to land first (this lane, board-free, one commit)

These are all "the registry says X, the source says Y". They cost an hour and they change what
everyone downstream believes.

| entry | what is wrong | correction |
|---|---|---|
| **R-12** | body says a 10-bit head and a 1024-node pool | The pool is **65536** nodes with a 16-bit head and `REVNODE_SENTINEL = 65535` — already corrected on 2026-08-27 in `ports/sqlite/build-sqlite-silicon.sh:943-949` against the resident bitstream's own source (`ariane_pkg.sv:587`, `capstone_rev_node.anvil:74,79`). Rewrite the body; the "will bite at call_dom at 1,060 splits" conclusion does not survive it. |
| **R-5** | header claims a live generic defect | All three named instances are reattributed: `C_GEN_CAP` was a **firmware** wedge and its fix landed; the `delin` case is archived R-2 (explained by C-13); the `scc`-derived load is the C-13 line. The generic residual ("a domain fault is a wedge") is tracked in **M-1** with a 2026-09-09 board observation. Close R-5 as SUPERSEDED, pointing at M-1 and the two archived entries. |
| **R-10** | close-out proposed demoting its secondary defect to "a one-line RTL observation" | **Do not.** `wt_axi_adapter.sv:196` (`is_cap_req = \|dcache_data.user`) and `wt_dcache_mem.sv:138` (`st_wr_cap = \|wr_user_i`) decide "holds a capability" by OR-reducing the metadata word and never consult `cap_type` — the same word-vs-granule confusion R-29 is about, live on `66c4e7517`. Link it to R-29 as a sibling account; R-10 stays OPEN, and its unrun stage-8 discriminator stays on the list. |
| **R-3** | header WORKED AROUND, body's last line says the root defect is live | The monitor still lacks the icache invalidate on domain switch; the workaround is now machine-enforced by preflight C15, which means **no boot can test R-3 any more** — the preflight refuses the staging that creates it. Correct the header to say so, and record that measuring it needs C15 deliberately disabled for one arm. |
| **`2026-09-05.tsv`** | row 200 (`sqslt1m`, sw44) carries `-O1` with the same image sha as row 94 (`sqm1`, `-O0`) | One of the two is wrong. Resolve from the image, annotate dated; until then `sqslt1m` is **not** an attributable S-04 arm. |

Also: file the helper's **EXPLAIN fault** as a new entry — a restored build define that faults the
domain at `SQ: E/share1` with cause 24 *before its own code runs* is a finding whether or not we keep
the define. Smallest reproducer is the helper's one-define build; root cause not claimed.

## 2. R-29 — separate, then fix (RTL lane; one bitstream; the lead flashes)

The audit returned PLAUSIBLE-BUT-UNPROVEN. Three accounts are alive and a fourth defect sits at the
same line, so a candidate written now would be aimed at a guess.

0. **Separation arms** (RTL lane), one per account, instruments that OBSERVE rather than infer:
   where the `sd` is at the `ldc`'s read cycle (store buffer / write buffer / array), whether the
   `ldc` hit or missed, and the `ldc`'s **own result register** — every reading so far is a readback
   after an `stc`, so `ldc` vs `stc` vs readback is undetermined. Predictions first, logs kept (the
   apart-PASS log was overwritten and now rests only on the records file).
   Accounts: the word-granular overlay at `wt_dcache_mem.sv:397`; the store buffer's word-granular
   disambiguation (`load_unit.sv:297`, `store_buffer.sv:279/287/293`); the miss-refill leg (`:354-358`).
1. **Fix candidate** where step 0 points. Whatever it is, it must **also** refuse the `.user` overlay
   for a non-capability entry — a plain store's wbuffer `.user` is provably zero
   (`store_unit.sv:363`), so a resident plain WORD-0 entry zeroes the high half of any `ldc` of that
   granule, and a word-1 term alone does not fix it.
2. **Lint = baseline exactly** (UNOPTFLAT 40, ANVIL 0). The file records three formulations of the
   tag-side term all taking 39 → 40 (`:384`). If the count moves, the candidate goes back.
3. **claim-auditor** on the diff; soft spot named: "does this cover the account step 0 selected AND
   the `.user = 0` overlay, and does it change `rd_ctag_o`'s cone?"
4. **R-24 rides this bitstream** (lead's decision): one line moving the FLU/DYN exception encoder from
   base 24 to the spec's base 23, `commit_stage.sv` already conformant. In the same commit, annotate
   the driver, the wedge tracer and the historical readings so an old `mcause 25` reads as the new 24.
   R-22's `stc` arm is a **measurement** — run it in simulation before the bitstream is defined, and
   batch its fix only if it turns into one. R-28 has nothing batchable.
5. **LEAD: flash decision** with the lint numbers, the sweep deltas, the §7 row.
6. **Acceptance boot**: `k800` → `s06copy` → `s06agg` (predicted **64**) → the distance ladder from
   §3 → closing set → R-25 probe last. Then R-29 archived, W-12 retired, mcause annotation verified.

RTL lane's estimate: half a day to a day to a candidate if the first formulation holds; sim pair +
88-row sweep ~1 h; auditor 20 min; synthesis ~1 h 45; add a day if the lint ring bites.

## 3. Board boot A — R-29 distance ladder and the re-triage (this lane, one boot, ~1 h)

Control first, ascending, the one that may hang last. Predictions written here before the boot.

| # | arm | question | prediction |
|---|---|---|---|
| 1 | `k800` | control | 4 |
| 2-4 | `s06agg_d1`, `s06agg_d2`, `s06agg_d4` — the S-06 kernel with 1, 2 and 4 filler instructions between the `sd` of `y` and the `ldc` | **the residency window on silicon**, which nothing has measured; this is the board's own version of the separation step | 66, then 64 at some distance; the step is the datum |
| 5 | R-15's 9216-byte capability-bearing global | attribution retracted 2026-07-31, never rerun on a fixed bitstream | returns |
| 6-7 | R-17/S-01 `uc` / `dp0` pair | never run in the 2026-09-05 sweep | both return; `dp0` last, it may hang |

Variants are a thin wrapper over `s06agg_kernel.h` via `LADDER_COMPUTE` and
`ladder_perf_domain.h` — distinct `DOMAIN_BASE_VA` each (preflight C15), `sha256sum` the set and
abort on any collision.

**Sequencing trap:** `uc`/`dp0` are rebuilt by `fpga-repros/S01-.../run.sh` through
`build-sqlite-silicon.sh`, which the SQLite branch changes. Rebuild them **before** merging §4, or
verify by hash against `IMAGE-HASHES.txt` afterwards — `SQLITE_FEATURE_SET=deployed` is *supposed*
to reproduce the old bytes, and that must be checked, not assumed.

**R-18 is not a board arm.** Its `rmB`/`rmC` geometry fits granule-scoped forwarding sharply, but the
RTL lane's first four arms are confounded by the same readback path the audit flagged and do not
reproduce the discrimination. It becomes a directed simulation arm once their instrument observes the
`ldc`'s own result.

## 4. SQLite stock-ness — the helper has delivered (this lane; audit, one boot, merge)

Branch `sqlite-stockness` now carries the seven-define restore set (`8a4096cdd65e`) and the
domain-side feature probe (`b4b944b2c8b1`). QEMU readings: deployed 0/6 marker `0x4EB00000`,
restored 6/6 marker `0x4EB00006`; SLT tallies identical both ways; silicon images
`e94fbc879cea` (1,445,016 B) and `c05b4729e3fa` (1,515,320 B), both two regions, both fit.

1. **Audit** against the primary sources: the six gated C APIs in `slt/check-feature-set.sh`, the
   `SQLITE_RESTORE` block pulled in at `build-sqlite-silicon.sh:937`, both harvests. Negative test:
   the checker must FAIL on the deployed image and PASS on the restored one — the probe's denominator
   is **six exported APIs, not seven defines**, and conflating them would misdescribe it.
2. **Board boot B** on the seven-define restored image: `k800`, then `select1.test`,
   `negative-control.test`, `slt_lang_aggfunc.test`, then the feature probe as the fourth stage.
   Predictions = the branch's QEMU readings. Separate boot from §3: one SLT domain per boot.
3. Fast-forward `dev`; `git show --stat` after.

## 5. Monitor and QEMU — one chain, in dependency order (this lane, board-free)

The scoping pass found **Q-07 and M-5 are one system**, which reorders them.

1. **Q-07 first.** `helper_csinit` at `capstone-qemu/target/riscv/op_helper.c:1198-1200` — three host
   `assert()`s that `SIGABRT` the emulator; all three map one-to-one onto spec exceptions
   (`cap-man-insn.adoc:415-421`), and the idiom to copy sits five lines away at `:728`. Ship `>`
   (spec and RTL both fault on `<=`); any `>=` relaxation is a spec change first.
   **`helper_csrevoke:920-921` must change in the same commit** — it was deliberately built to leave
   `cursor = end` to feed QEMU's own `csinit`; align it to the RTL's `cursor = start` or the fix is
   self-inconsistent. Gate: `run-linear-uninit-corpus-probe.sh` (its expectations need rewriting with
   the fix) plus `run-nullblk-all.sh`; both, together, serialized on the rootfs lock.
2. **M-5 becomes reproducible the moment Q-07 lands** — nullblk goes red and stays red until M-5 is
   settled, which is the correct order, not an obstacle. Sites: `sbi_capstone.c:1196-1197`, **and a
   second one the registry never mentions** at `:1340-1341` in `share_child_region`. Both must change.
   The work is a decision, not a patch: no legal sequence takes UNINIT(cursor=start) to LIN on the RTL
   except the `CAPTYPE` debug op, so the choice is amend the spec's `INIT` precondition, add a
   monitor-side reclaim that avoids `INIT`, or change what `revoke` leaves behind in RTL.
3. **I-03** is neither monitor nor QEMU. The high-value fix is a runtime guard in
   `benchmarks/rv8/adapted/umm/umm_malloc.c` — `umm_multi_init_heap` at `:296` already has a run of
   early-exit guards at `:307`/`:312`; a third on `((uintptr_t)ptr & 15)` drops in beside them.
   Today the invariant is enforced by **six ad hoc `sed` rewrites** in the BEEBS build scripts and
   nothing catches the seventh benchmark. Prefer the runtime guard over `_Static_assert`, whose
   support in the domain compiler is unverified. Gated by the nightly benchmark suites, not by any
   runtime-qemu probe; a green QEMU run proves nothing here.
4. **M-1 is smaller and different than the registry says.** The trap-vector slot write already exists
   at `sbi_capstone.c:1024-1025` but is behind `#ifdef CAPSTONE_DOMAIN_TRAP_VECTOR`, which **is defined
   in no build file** — so "on silicon since 2026-09-02" means tested once under a one-off build, not
   shipped. Turning it on only moves the wedge to `_cap_trap_entry+8`: a trap does not switch domains,
   so the handler runs in the *domain's* capability context and `CINCOFFSETIMM` on the domain's
   cscratch is an unexpected operand. The live firmware item is a **domain-safe prologue** at
   `sbi_capstone.S:13-16`; the rest is RTL and ownership moved to the RTL lane on 2026-09-09. Verify
   with `run-smoke.sh` + `run-shared-region-probe.sh` as a don't-break-the-monitor guard only —
   QEMU delivers domain faults cleanly and cannot reproduce M-1 at all.

**Every monitor edit touches four identical source copies** (`caplifive-buildroot` and
`caplifive-system`, each `package/` and `components/opensbi/`). The `caplifive-system-dev` copies are
stale — leave them.

## 6. The standing four, best-first (lead: "do best one, or all of them maybe")

1. **C-5 — lift the 4 KiB code window.** Highest value: it is what full CoreMark and Dhrystone need,
   i.e. paper-bearing numbers. One constant in
   `capstone/tests/runtime-qemu/gp-free-domain/link-gpfree.ld`, already QEMU-validated at 16 and
   32 KiB and silicon-validated at 32 KiB. Consumers to re-check: `start-gpfree-captable.S`,
   `build-ladder-domain.sh`, `multi-tu-slot-collision.sh`. **This is the one if only one.**
2. **R-12 — correct, then measure.** The body correction is §1. Then the open question: the board
   printed `rev-node head = 65047 (99.3% of 65535)` in sw45 *and* sw47 — the identical value in two
   boots with different stage sets, read after a wedge, from a path that can serve cached values. It
   is either a live near-exhaustion risk or an instrument artefact; one read of the head at two known
   points in a healthy boot separates them. Do not treat 99.3% as a finding until it does.
3. **I-02 — adopt the convention.** Allocate by checking **both** `ISSUES.md` and
   `git log --all --grep`, backed by a committed ID ledger, so an ID assigned only in a commit
   (C-25 is the live example) cannot be handed out twice.
4. ~~**C-38 — rename the register-form `CAP_CALL` mnemonic**~~ **DONE 2026-09-10** (compiler lane,
   `8b2544a6c3e9`), by parser precedence rather than a rename — the disassembler already prints
   `call a0, a1`, so a rename would have had to change both sides of the round trip. Lit 102/102.
   Spun out: `call a0, foo` still does not assemble, filed as **C-45**, never worked, no consumer.

## 7. Registry disposition, revised

Only **R-5** closes on evidence (SUPERSEDED, §1). **R-4** is a records decision, not an evidence one:
one 2026-07-28 prose observation on a bitstream three reflashes old, no reproducer, and the sweep
itself said "keep OPEN" where the close-out said "close" — closing it means closing on the absence of
evidence, which is the trap this project keeps falling into. Recommendation: retitle it
RECORD ONLY with that stated plainly, rather than either closing or leaving it looking actionable.

**R-10, R-3, R-11, S-04, S-10/S-10b all stay open**, each with its blocker now named (§1 for R-10 and
R-3). R-11 gets one cheap off-board job: `check-repr.py` exists and runs, but on the current corpus
every domain reports `tot = 2097152` — exactly the window — so **the truncation branch never
executes**. Run it on a refreshed corpus *and* give it a positive control past 2 MiB; a clean exit
from a check that cannot fire is not a result. S-04 needs a second draw plus the stage-164 bit read
on the board, N=1 today.

The rest of the open list is unchanged: RTL lane holds R-22/R-21/R-13/R-19/R-24/R-28/R-29; the
compiler lane holds C-43 (land `5d2932a9`), C-14, C-4 (no status token — needs a human read), C-17
(latent).

## 7b. Progress, 2026-09-10 (all pushed)

| item | state |
|---|---|
| §0 lead | `push-final.sh` run; all four nested remotes verified in sync |
| §1 corrections | DONE. R-12 rewritten (16-bit head, 65536 pool, exhaustion DEADLOCKS rather than wraps, SQLite scare void) and its 99.3 % reading WITHDRAWN — every healthy boot reads the sentinel, only wedged boots give 65047, and the sentinel cannot be the true head or no domain would run. R-5 closed SUPERSEDED. R-10 corrected TWICE: its cited sites are repaired, but the OR-reduce pattern survives on the REFILL path at `wt_dcache_mem.sv:358`/`:501` → PARTIALLY FIXED. R-3's header now says the root defect is live and preflight C15 makes it unmeasurable. S-14 filed for the EXPLAIN fault. tsv opt-level labels corrected. |
| §2 R-29 | Mechanism SEPARATED: the load MISSES and takes the refill leg while the entry that would repair it is resident but invisible to a word-gated overlay. Store-buffer and `.user = 0` accounts both REFUTED by traced arms. **Board sw49: the silicon window is ONE instruction wide.** Still owed: `wbuffer_hit_oh`/`wbuffer_be` read directly. No fix candidate yet; R-24 rides the same bitstream. |
| §3 board boot A | DONE as sw49 (the distance ladder). R-15 and R-17/S-01 arms NOT run — they moved out when the ladder became the higher-value use of the boot. |
| §4 SQLite | Helper delivered the seven-define set and the probe. Audit complete: the feature-set gate verified two-sided by EXIT STATUS in all four combinations; budgets tie out. Board boot sw50 running. |
| §5 monitor/QEMU chain | NOT STARTED (Q-07 → M-5 → I-03 → M-1). |
| §6 standing four | C-5 CLOSED as stale — the window is a per-image knob discovered from the ELF, positive-controlled by building one rung at both windows. R-12 done (§1). I-02 done: `tests/next-issue-id.sh`, positive-controlled on C-25. C-38 DONE by the compiler lane; C-45 spun out. |

**The lead's open questions, answered 2026-09-10 (all committed).**

| question | answer |
|---|---|
| R-4: close or keep open? | **RECORD ONLY.** Closure as "not reproducible" is rejected as overstating the record — nobody ever attempted reproduction, and the sweep marked it UNTESTABLE because there was nothing to run. Its symptom class is now owned by R-19, R-10-secondary and R-29, so a new sighting goes to whichever it matches. No attribution is made: R-4 has no artefact and a fit is not a mechanism. |
| C-4 disposition | **FIXED, both halves re-verified.** Verified against the sub-entries: the "remaining domain-creation bug" the heading promised does not exist. The proposer's caveat (recorded evidence, rungs not re-run) is kept. Could not be moved to the archive — see the scan note. |
| C-14 disposition | **RETRACTED AND REDONE.** First answer said "attribution pending Q-04"; an audit showed the dependency was manufactured (the entry's own v3 box says the compiler attribution was never in doubt) and the defect is already FIXED — `copyPhysReg` branches on register class, so scalar copies are an ALU move. Two residuals rehomed: the untagged-in-GPCR live copy to **C-32**, the MOVC modelling defect to new **C-46**. |
| C-45: land this cycle? | **Yes**, as its own commit on top of C-38's and never squashed with it. The code and its test are open now; the cost of leaving it is a declared instruction form the assembler silently refuses; the risk is bounded by the same file, same test, 102/102 suite. |
| Scan: stop reading removed lines? | **Yes, and context lines too — WARN, not BLOCK.** Full reasoning, patch and six required controls in `precommit-scan-removed-lines-proposal.md`. NOT applied: it is a release gate and therefore the lead's. Hit three times in one session; it is currently preventing C-4 from being archived. |

**⚠ Q-04 was ruled and the ruling is RETRACTED the same day.** It argued the spec was explicit because
*"`NOT_CAP` is type 0, and `0 != 1`"*. **The spec has no `NOT_CAP` type** — its table is Linear 0,
Non-linear 1, Revocation 2, Uninitialised 3, Sealed 4, Sealed-return 5, and `NOT_CAP = 0` is the RTL's
enum, which inserts it at zero and shifts everything up. The RTL's own comment says so. The syllogism
evaluated a spec sentence with RTL constants, which is the mistake that cost boot sw39 and that a
memory note exists to prevent.

The ambiguity the ruling denied is real: MOVC's operands are annotated as capabilities, and the spec
commit that removed its *"is not a capability"* exception left the consumption clause untouched — so
scalar-exemption is a RESTORATION of that clause's original precondition rather than an amendment,
which is the opposite of what the ruling claimed and makes it the CHEAPER option. **Q-04 is a spec
question and remains the lead's.** What survives is only that the RTL does null a NOT_CAP source, which
is board-confirmed.

## THE LESSON OF 2026-09-10, and it is worth more than any finding in this plan

**Six instruments across two lanes produced readings without ever creating the condition they were built
to test.** In one night, on one investigation:

1. **The board driver's classifier** keyed on the SQLite host's markers, so an `rtpc` stage could only
   ever read "domain never created" — it had been misreporting every probe of that shape.
2. **`s06agg-shape`'s first version** put the store four instructions from the load. It passed on three
   RTL revisions and was reported as exonerating the hardware.
3. **`r29-lowword`** was offered as the discriminating arm. The mechanism it was built to test predicts
   it should FAIL; it passed because the write-buffer entry was never resident.
4. **Both R-29 separation arms** inserted a load that brought the line in, so the wide load HIT and
   never took the refill leg — removing the very condition under test.
5. **The `lsugate` probe (mine), TWICE, for two different reasons.** First staged under the wrong
   host, so the region transfer never arrived and the domain never ran — **the driver printed `ran=0`
   and the parser read the retval anyway**. Re-run with the tracer on, it wedged at image offset 0xb0,
   which is `delin gp` in the **entry glue**, roughly 0x280 bytes before its own measurement. Two
   boots, no measurement, and on the second I briefly recorded the wedge as evidence against a
   hypothesis before reading where it actually happened. **A wedge that occurs before the experiment
   is not evidence about the experiment.**

6. **`host-sweep.sh` itself.** It reuses compiled ELFs from a reference worktree `wt-ref` when the
   `.S` is unchanged — and **`~/dev/llvm-capstone-rebuild/wt-ref` does not exist** (verified; its
   siblings do). With no reference tree every simulation runs to the timeout, so the script emits a
   full set of plausible `TIMEOUT 400013` rows rather than erroring. Nine tests were read that way,
   including one that passes in the container path in seconds on the same model. **A missing
   dependency presenting as uniform results.**

Five of the six had no warning to discard. The one that did is the most useful of them: **the check
existed and was simply not wired to the verdict.** And the sixth is the worst class of all — the
instrument was not merely unguarded, it was *not connected to anything* and still produced a full table.

**The one arm that actually refuted something** — `r29-sep-userzero-miss` — is the one whose reading
contains its own positive control: a correct LOW word at the failing cycle, which only the write buffer
could have supplied, proving the entry was resident exactly when the account said it was invisible.

### The rule (RTL lane's wording, adopted here)

> **For every arm, name IN ADVANCE the observation that proves its condition existed, and make the
> parser REFUSE a verdict without that observation rather than score the run.**

Applied concretely, and each of these is now in the corresponding instrument:
* `lsugate` — refuse any reading whose RESULT line says `ran=0`; refuse a wedge as a verdict without a
  latched `mcause`; **and check the latched `mepc` against the measurement's own offset before reading
  the wedge as the measurement at all** — the second attempt wedged 0x280 bytes short of it.
* `r30-fill-init` — refuse unless `LCC` shows the cursor actually reached `end` (bounds can widen on a
  register writeback, so "four stores into a 64-byte region" does not imply it).
* the R-29 arms — refuse without `wr_cl_vld`, which is what says the load actually missed.
* `host-sweep.sh` — exit non-zero when `REFWT` is absent, instead of sweeping against nothing.

This is a **sharpening of** the existing rule that a clean result is not evidence until the check is
known to fire, not a replacement for it. The addition is that the proof-of-condition must be **named
before the run and enforced by the parser**, because every one of the five above was written by someone
who knew that rule and believed their arm satisfied it.

**Proposed for `CLAUDE.md`** as one sentence under "A CLEAN result is not evidence until the check is
known to fire" — not added, because that file is the lead's:
*"Name the observation that proves the triggering condition existed, and make the instrument refuse a
verdict without it; five arms in one session scored runs whose condition never occurred."*

## STATE AS OF 2026-09-10 NIGHT — read this section before the ones below it

Four things I recorded earlier today were **retracted the same day**, three of them by adversarial
audit. The entries carry the retractions in place; this is the summary.

| I said | corrected to |
|---|---|
| Q-04 is settled by the spec text | **RETRACTED.** The argument mixed two type-numbering systems (the spec has no `NOT_CAP`; its `0` is Linear). Q-04 is a genuine spec question and the lead's. |
| C-14 is gone-on-silicon with attribution pending Q-04 | **RETRACTED.** The dependency was manufactured and the defect was already FIXED. Two residuals rehomed to C-32 and new C-46. |
| R-30's fix is a conformance fix | **RETRACTED.** The spec says `cursor <= end` is illegal, so the pre-fix RTL was CONFORMANT. The change is a deliberate **spec deviation** and needs the spec amended with it. |
| Fixing R-30 closes Q-07 | **RETRACTED.** It NARROWS it. Post-fix the RTL accepts `cursor >= end` while QEMU asserts `== end`. |
| C-46's fix is to tie `rd` to `rs1` | **WITHDRAWN.** That would break every capability copy. |
| M-5 has two consumer sites | **Five.** The widest, `split_out_cap`, needs no re-share at all. |
| R-31's fix closes the disclosure | **Necessary, not sufficient.** A plain scalar load is checked only by a block measured INERT in our domains in 2026-08, and QEMU's revoke leaves the cursor at `end` so the shortcut still succeeds there. |

**What survives all of it:** R-30's arithmetic (a full fill leaves the cursor at `end`, so INIT is
unreachable by filling) and R-31's polarity (the RTL returns LINEAR where the spec says UNINIT), both
demonstrated by directed tests with negative controls that fail on the unfixed tree.

**So the lead now has a SPEC DECISION, not just a flash decision:** whether `end` is inclusive or
exclusive. The surviving argument for exclusive is that QEMU is exclusive throughout and the whole
software stack is validated against QEMU. The spec itself leans inclusive wherever it speaks, and the
RTL is split against itself. Nothing should be synthesised until that is answered, because the RTL is
in flux until it is.

**A fourth arm produced a reading without creating its condition, and it was mine.** Boot sw51's LSU
sufficiency probe was staged under the ladder host instead of the transfer host, so the region share
never arrived and the domain never ran. The driver printed `ran=0`; my parser matched only the retval
and reported the zero as a measurement. One wasted boot, re-running with the host fixed and a parser
that refuses `ran=0`.

**2026-09-10, later: answering M-5 uncovered two larger defects, and the bitstream plan changed.**

M-5's "design choice" rested on a false premise, and reading the flashed RTL to settle it produced
**R-30** (`INIT` is unreachable by filling ANY UNINIT region — the cursor tops out at `end`, `INIT`
demands `> end`, shortfall one byte) and **R-31** (REVOKE's permission clause is inverted against the
spec, so revoking a linear borrow of a WRITABLE region returns a readable LINEAR capability instead of
UNINIT-at-base — the reinitialisation step is skipped and the borrower's data is disclosed). Both
verified independently by two readers against `66c4e7517`, including the lattice-operator reading that
would have collapsed R-31 had it gone the other way. Neither is demonstrated by execution yet, and both
entries say so.

Neither is a regression: the same arithmetic and the same clause are on the previous bitstreams, and
R-30 is in the spec's own text. Nothing here argues against the 2026-09-09 flash or the results it
produced.

**This reorders the bitstream into two builds:**

| build | contents | state |
|---|---|---|
| first | R-30 + R-31 | branch `r30-r31-init-revoke`, **lint CLEAN at baseline 40**, predictions written before the run; functional arms and the auditor outstanding |
| second | R-29 | candidate works functionally but takes UNOPTFLAT 40 → 41; **the lead's fork**: spend a synthesis on the S-10 precedent, or reformulate first |

R-29 must not hold up R-30/R-31 — a disclosure gap and a dead type outrank a one-instruction window
that already has `W-12` in force. **R-31 must not ship without R-30**, or a silent disclosure becomes a
live monitor trap on every RW revoke.

**One decision is genuinely the lead's and it gates the first build:** declare whether `end` is
inclusive or exclusive. Recommended EXCLUSIVE — every RTL access path and all of QEMU already assume
it, it makes the fix a one-character change, and **it also closes Q-07**, whose current state is that
QEMU and the RTL accept disjoint `INIT` operands.

**A process failure worth the lead's eye.** One commit tonight went in while `precommit-scan` said
BLOCKED, because the command printed the scan's exit code instead of gating on it. The hit was a false
positive and the content was verified clean, but the procedure was wrong and it is the same shape as
the documented "never filter between a gate and its exit status" failure. Gating is restored.

Two properties of the gate came out of it, the second corrected by the compiler lane after I first
described it wrongly. **(1)** The pattern at `precommit-scan.sh:163` is
`\btoken\s*[=:]\s*\S` — the word "token" followed by a colon or equals and any non-space. Backticks
are irrelevant; the fix is to avoid that construction entirely and write "status word" or name the
value directly. **(2)** The script appends `git diff` output wholesale (`:58`, `:60`, `:67`) with no
filter for added lines, so **REMOVED lines are scanned too** — once such a phrase is committed, the
later commit that deletes it is blocked as well. Cleaning up a false positive therefore needs either a
gate change or a bypass, both the lead's call, so the wording was left alone. The pattern guards the
real console credential and should not be weakened.

## 8. Order

Day 1: §1 corrections, §6.1 (C-5), the R-29 folder split the lead approved, §3 board boot A, §4 audit
and board boot B, merge. RTL lane starts §2.0. Day 2: §5 chain (Q-07 → M-5), §6.2-6.4, R-22's
measurement, the R-29 candidate through lint and audit, synthesis overnight. Day 3: flash decision,
acceptance boot, R-29 and R-24 recorded.

"Finished" = every open entry fixed, closed with its evidence, or parked with its blocker named in
its own header.

## 9. Verification

- Every board arm's prediction is in this file before its boot; readings come from the run's own
  transcript segment into `board-results/*.tsv` **with the image hash**, and any result is cited by
  hash, never by label (CLAUDE.md, added 2026-09-09).
- A boot whose control fails is VOID. At most one expected-to-wedge arm, last.
- R-29: separation arms with kept logs → candidate → sim pair + 88-row sweep identity → lint at
  baseline → auditor → synthesis row → board acceptance 66 → 64.
- Q-07/M-5: `run-linear-uninit-corpus-probe.sh` and `run-nullblk-all.sh` together, under the QEMU
  lock; M-5's two sites both changed.
- R-11: the checker must be shown to FAIL on a past-2 MiB positive control before its pass counts.
- Every commit scanned with `precommit-scan.sh` by absolute path, `-o` own paths, `git show --stat`
  after; nested pushes through `push-final.sh`.

## 10. Lead decisions

**Taken:** R-29 splits into its own folder now; R-24 rides the next bitstream with the mcause
annotation; the four standing items are in, best-first.

**Still yours:** the flash decision after synthesis (§2.5); whether R-4 is retitled RECORD ONLY or
stays open (§7); whether the silicon default moves for S-04; and the M-5 design choice in §5.2 if the
answer is "amend the spec".
