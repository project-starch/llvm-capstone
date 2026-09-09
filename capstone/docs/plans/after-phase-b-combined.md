# Amendment 2026-09-09 (evening): status, the R-25/26/27 board track, and the registry split into two files

*(Amends the combined plan below, which the lead approved this morning. Three things changed: the CHERI smoke
and Q-06 are done; the RTL lane's R-25/26/27 state is now known precisely; and the lead's new rule for the
registry: **`ISSUES.md` holds only the open, relevant issues; every resolved entry goes to a separate archive
file.** That replaces section D.2 of the plan below.)*

## 1. Where things stand now

| item | state |
|---|---|
| CHERI smoke (build B) | **PASS on silicon** 2026-09-09 14:48–14:54, non-volatile route, Capstone restored and verified; records `~/capstone-artifacts/cheri-cva6/board-run-1/` |
| Q-06 | **FIXED, QEMU-validated** (monitor 91c48f3, wrapper 6058091, buildroot 4244cbf, caplifive-system 8616741, unpushed): null-blk suite 3/3 for the first time; tier on the fixed images running; the FPGA build owes ONE control boot (new global shifts two FPGA-only UART globals by a slot; `.c.S` diff enumerated) |
| R-26 | fixed in RTL, sim-verified at the verified 40-cycle latency; on branch |
| R-27 | found, drain fix sim-verified (58-arm set: 57 PASS + the deliberate control FAIL, 0 hangs); on branch |
| R-25 | fixed in RTL, self-checking test PASS; on branch |
| RTL Phase 3 (two/three final commits on `fpga-testing-dev`) | **pending** — then the **LEAD pushes** `fpga-testing-dev` (hook blocks the branch name) |
| RTL Phase 4 (synthesis, ~3.5 h) | not started; prediction written (WNS −11.7…−15.3 ns, 168.9–170.5 k LUTs) |
| Board Phase 5 | pending on the pre-flash boot (below) and the **LEAD's flash decision** |
| Registry | 6427 lines; 49 entries above the in-file "Archive" line of which **19 are fixed**; 50 below of which **10 are OPEN** (I-03, I-02, C-43, C-37, C-38, R-19, R-18, R-21, R-22, R-24) — mixed both ways, which is the lead's complaint |

## 2. The R-25/26/27 board track (this lane executes; two decisions are the lead's)

> **Status 2026-09-09 15:40.** Pre-flash boot **sw39** ran on the Q-06 firmware (fw_payload aa471de2e4e8):
> k800 + six BEEBS rungs 7/7 at the oracles — the Q-06 control boot is done, zero fault tags. The two R-25
> domains (r25same control, r25dup probe; sources in `tests/runtime-qemu/silicon-ladder/`) were built from the
> RTL lane's test construction (LIN region, cursor past end, in-place CAPTYPE to UNINIT, then INIT), because
> the spec and the RTL accept INIT only for cursor > end, which no store sequence and no revoke produces. On
> silicon the control wedged at its INIT with mcause 27 (UNEXPECTED_CAP_TYPE). Cause, from the RTL lane: the
> RTL's type numbering is not the spec's (asm_insn.h: NOT_CAP 0, LIN 1, NONLIN 2, REV 3, UNINIT 4 …); the
> domain wrote 3 = REVOKE, and INIT rejected it as it must. CAPTYPE itself works in a domain (no mode check;
> in place on rd; rs1's low three bits are the type). A domain fault is not delivered to the monitor on this
> RTL (M-1's open half), so the core wedged and the probe never ran: both R-25 arms VOID for sw39. Rebuilt
> with UNINIT = 4; rerun as boot sw41 (k800, r25same, r25dup last) after variant D (sw40). The QEMU side
> cannot stand in: QEMU does not decode CAPTYPE and its INIT accepts exactly the operand the RTL rejects
> (Q-07). **sw41 (16:03): R-25 CONFIRMED on silicon** — r25same 0x25000001, r25dup 0x25000001 (the store
> through the INIT source landed: the duplicate is live), k800 = 4, zero fault tags. **sw40/sw42: variant D
> CLEAN** on the current silicon (sw40 with two of the three CCSRRW fence.i dropped, sw42 with all three;
> 7/7 at the oracles each, zero fault tags), so the post-flash D/E boots are no-regression checks and the
> R-26/R-27 evidence stays the simulation. Synthesis of 66c4e7517 waits for the lead's word to the synth
> lane in its own session (~1 h 45 min from the go). Two registry entries filed from this work: Q-07 and
> M-5; one line for the R-25 folder later: the
> RTL does not implement the spec's cincoffset-past-end rule (cap-man-insn.adoc:262), which is what makes
> the construction possible.

1. **Now, off-board (this lane):** build the pre-flash batch — the R-25 domain probe (INIT `rs1≠rd`, probe
   `rs1`, marker 1 = duplicate present / 0 = NOT_CAP; the RTL lane owes the `.S`/image or its spec, else I write
   it from the registry's R-25 entry), firmware variant **D** (drop the three CCSRRW-adjacent `fence.i` in
   `sbi_capstone.S`; `fence-variant.py` gains a `D` mode; QEMU byte-identity is irrelevant — FPGA-only sites),
   the I-4 probes (`accum_probe`/`accum2_probe`, rebuild from the sweep's recipe), the C-3 RV8 -O2 rung, and
   the Q-06 firmware as the baseline of that boot. Entry VAs distinct; preflight records for the `|label`
   rungs; oracles from the host values.
2. **Pre-flash boot sw39 on the current silicon** (one boot, control first, one unknown last): k800 → the six
   BEEBS rungs (the Q-06 firmware's control set) → I-4 probes → C-3 rung → the R-25 probe (predicted **1**)
   → variant D LAST (predicted CLEAN by the RTL lane's reading; if so the post-flash D/E boots are
   no-regression checks, written down before the flash). This is also Q-06's owed control boot.
3. **LEAD:** push `fpga-testing-dev` when the RTL lane reports Phase 3 done; the RTL lane synthesises (3.5 h).
4. **LEAD:** flash decision with the lint numbers, the sweep deltas, the §7 row and the sw39 readings.
5. **Post-flash boots** (this lane): boot 1 = the closing set on the new bitstream (k800, six rungs, SLT
   select1, transfer probe); boot 2 = R16 acceptance, the S-06 trio, the R20 rung, the R-25 probe (predicted
   **0**), variants D and E. Then the `fence.i`-drop monitor commit and the registry lines R-25/26/27 →
   FIXED ON SILICON (through bench while it holds the file).
6. **Re-triage boot** on the new bitstream: R-17/S-01 pair, R-18 frozen images, the resolved-but-retained
   packages as regression tests.

Timeline: 1 today; 2 tomorrow morning (~1 h board); 3–4 the RTL lane's schedule plus the lead; 5 the day of
the flash (~2 h board); 6 the day after.

## 3. The monitor and compiler buckets (this lane, unchanged order; Q-06 done)

> **Status 2026-09-09 16:30.** M-4 DONE (monitor 0658243: `call_domain_with_cap` bounds `dom_id`; generated
> files differ only in that function on both targets; smoke, borrow-cost, null-blk unchanged). M-3 DONE
> (monitor bff5b71: the ecall trampoline reports error = -1 when a handler's value is -1; generated files
> byte-identical to M-4's; smoke, five probes, null-blk, M-2 control, hole check, cascade unchanged). I-5
> CLOSED on evidence (every monitor error site already names itself on the UART; entry archived). M-1's
> remaining half is RTL-side (sw39 evidence written under the entry; ownership to the RTL lane). Both
> monitor changes owe one FPGA control boot, batched into the first post-flash boot; the nightly tier on the
> M-3 images (M-4 + M-3 + Q-06) is **18/18** (2026-09-09 16:30–17:15, no suite under load). **Q-04 is NOT a
> QEMU fix for this lane:** the C-14 trail's standing position is that scalars are probably exempt from
> MOVC's nulling and the RTL's behaviour is the oversight, so Q-04 needs a spec ruling from the lead and
> the RTL lane; if the ruling goes the other way it becomes C-14's compiler fix. **C-37 / C-9 / C-43 belong
> to the compiler lane** (llvm/ and codegen are theirs by the session split; §C below named this lane by
> mistake) and have been handed over. This lane's monitor bucket is therefore empty until the post-flash
> boots.

M-4 (bound `dom_id` in `call_domain_with_cap`) → M-1 (read `RTL-domain-trap-vector-unset/` first: its
firmware half is already on silicon; what remains may be RTL-side, in which case M-1 moves to the RTL bucket)
→ M-3 (real SBI error returns; the module's failure paths go live, module suite must stay green) → I-5 → Q-04
(QEMU MOVC nulling; grep for MOVC-of-scalar shapes first) → C-37 → C-9 → C-43. Each its own commit, `.c.S`
gates on both targets, the FPGA control boots batched into the boots of section 2 (sw39 takes M-4/M-1 if they
land before it; otherwise post-flash boot 2). About three working days for the monitor items, two to three for
the compiler items.

## 4. The registry split (replaces D.2; bench executes on `archive-docs`; this lane reviews the diff)

- **Two files in `docs/ref/`:** `ISSUES.md` (title stays "Open issues registry"; contains ONLY entries whose
  header status is OPEN / CHARACTERISED / WORKED AROUND / LATENT / RECORD-ONLY-but-open) and a new
  `ISSUES-ARCHIVE.md` ("Resolved issues — kept for provenance"; contains every entry whose header status is
  FIXED / CLOSED / RESOLVED / RETRACTED / GONE / NOT A BUG / SUPERSEDED / EXPLAINED / NOT REPRODUCIBLE /
  RECORD ONLY (folded) / RESERVED). **The rule is the header's status token, never the entry's current
  position** — 19 fixed entries sit above today's archive line and 10 open ones below it.
- **Mechanics:** entries move VERBATIM (heading, body, strike-throughs, retraction boxes), one `git mv`-like
  cut/paste per entry; the in-file "## Archive — fixed, kept for provenance" section disappears from
  `ISSUES.md` (its preamble becomes the archive file's preamble); `## How to add an entry` stays in `ISSUES.md`
  and gains one sentence: "when an entry's status becomes final, move it to `ISSUES-ARCHIVE.md` in the same
  commit; IDs are never reused". Both files carry a one-line pointer to the other at the top. The
  "Last updated" line is refreshed. The 21 headers my parser could not classify are placed by a human read of
  the status token (bench has classified them already).
- **Cross-references:** no script parses `ISSUES.md` (checked: only comments in `run-nightly.sh`,
  `run-q03-region-hole-check.sh`, `m2_region_overflow_host.c`, ladder kernels, CLASSIFICATION.tsv). Prose
  references of the form `ISSUES.md:<line>` are already stale by construction and stay. References that name a
  fixed entry's HEADER as a source of truth (the S12 repro README: "the S-12 header in docs/ref/ISSUES.md is
  the source of truth") are updated to `ISSUES-ARCHIVE.md` in the same commit (`grep -rn "ISSUES.md" capstone
  --include=*.md` lists them; expected under ten).
- **Ordering inside the archive:** by ID family then number (R, S, C, Q, M, I, F), so an ID is found by
  scanning; the archive file's preamble says entries are frozen and corrections go under them dated, as now.
- **Review gate before the fast-forward:** a script check that every ID present before the split is present
  exactly once across the two files (`grep -hoE '^#{2,3} [A-Z]-[0-9]+' both | sort | uniq -c`), that no open
  status token appears in the archive file and no final status token heads an entry in `ISSUES.md`, and that
  the two files' total entry count equals the old file's. I run it on bench's branch before the
  fast-forward; bench does not fast-forward until I say so.
- Registry lines that land while bench holds the file (Q-06 FIXED, R-25/26/27 updates) go to bench as text
  and are written into whichever file the status now dictates (Q-06 → archive at the split, with its board
  control-boot caveat carried in the header).

## 5. Verification for this amendment
- sw39 rows in the tsv from the run's own transcript; predictions for D and the R-25 probe written into the
  plan doc BEFORE the boot; a boot whose control fails is VOID.
- Q-06: tier PASS on the fixed images (or its FAIL explained the way the 5A tier's was), then the dev gitlink
  commit and the nested pushes through `/tmp/capstone/push-final.sh` (the lead's credential).
- The split: the three-way count check above, `git show --stat` shows only the two registry files plus the
  handful of cross-reference edits; every commit scanned.

---

---

# Combined plan 2026-09-09: the 33 open issues, the R-25/26/27 bitstream on the board, the CHERI smoke, archiving

*(Board lane. Approved by the lead 2026-09-09 as written, with the session split (board: A, B, C and every
boot; bench: D and the registry re-triage). The RTL lane's own plan is in their fix-cycle history note; the
earlier `after-phase-b.md` is executed except its R-26 half. Decisions that are the lead's are marked
**LEAD**. Nothing here edits CLAUDE.md.)*

## Context

The lead asked four things: where the registry is, whether `fpga-repros/` and the docs carry resolved
material that should be archived, whether the CHERI lane's volatile program + smoke test can run now, and
one plan covering the 33 open registry entries, the R-25/26/27 validation the RTL lane is preparing, and
the CHERI smoke — with an honest "when are we finished".

Facts gathered read-only today:
- The registry is `capstone/docs/ref/ISSUES.md` (one heading per ID; `## Q-06`, `### R-26`, …). My count:
  96 headed entries, 42 with a fixed/closed status, 33 open, 21 whose header line my parser could not
  classify (mixed strike-through statuses — they are not unknown work, they need a human read once).
  17 fixed-status entries still sit ABOVE the file's own "Archive — fixed, kept for provenance" section
  (S-12, S-02, S-03, S-05, S-07, Q-03, Q-05, Q-01, S-08, S-06, C-18, R-1, C-2, C-19, M-2, C-15; R-26 now too).
- `tests/fpga-repros/` has 18 active folders and an `ARCHIVED/` folder whose README rule is "fixed in
  silicon AND verified on the board → `git mv` + banner + index row; kept as bitstream regression tests,
  never handed over as open". By their own 2026-09-07 banners, these qualify now: R01 (R-1 GONE on
  5097eb166), R16 (RESOLVED, kept as acceptance test), R20 (fix present by content), S06 (FIXED, acceptance
  passed), S12 (fixed in the flashed RTL), S08 (bitstream-specific, superseded). NOT qualifying: R18
  ("not reproduced with frozen images" is not fixed), R19, S01, S07 (residual not observed — the registry
  says FIXED, the folder says "not observed"; read both before moving), S09/S10/S11/S13, the three RTL-*
  observation folders.
- Docs: `docs/plans/archived/` exists (21 there, README index). Plans whose own status says done:
  `monitor-unification.md` (Phase A+B complete), `capstone-column-xlang.md`, `xlang-phase1-followups-TODO.md`,
  `instrumentation-cleanup.md` (closed). `docs/ref/SILICON-BLOCKER.md` is marked SUPERSEDED by `docs/README.md`.
  Everything else in `ref/` and `design/` is reference or design and stays; `history/` is append-only.
  (A grep for "superseded/DONE" in file heads produced false hits — ISSUES "closed", fpga-user-manual
  "Done", RATE-RULE — those are NOT candidates.)
- The board is idle on this lane's side (sw38 done). The next Capstone board runs wait for the RTL lane's
  bitstream. The CHERI package (`~/capstone-artifacts/cheri-cva6/board-package/`) is audited and revised:
  volatile program only, restore by power cycle.
- The RTL lane's status (their section below): R-26 and R-25 fixed in sim; R-27 found, drain fix
  sim-verified; Phase 2 on the combined worktree; synthesis next; **LEAD** decision pending whether R-27
  joins the bitstream (their recommendation: yes; already put to the lead — not re-asked here).

## Where things stand (progress)

| bucket | open | disposition from the evidence on file (`plans/bug-sweep-2026-09.md` rows) |
|---|---|---|
| close on existing evidence | 6 | R-21 (GONE at 5097eb1 for cincoffset), R-6 and R-8 (their rungs pass on s12fix), R-4 and R-5 (no reproducer, none produced by the sweep), C-5 (a documented design limit, not a defect) |
| needs ONE board reading | 4 | I-4 (probes returned zeros — rerun k800 + accum probes), R-17/S-01 (the uc/dp0 perturbation pair), C-3 (RV8 -O1/-O2 board line never run), R-11 (`check-repr.py` on the current corpus, no boot) |
| real fixes, this lane, board-free | 9 | Q-06, Q-04, M-1, M-3, M-4, I-5, C-9, C-37, C-43 (C-17 latent, no work until it bites) |
| RTL lane | 6 | R-25, R-26, R-27 (in flight), R-28 (needs a directed test), R-22 (the `stc` arm never run), R-19 sim half |
| **LEAD** decision | 4 | C-38 (mnemonic naming), I-02 (ID allocation convention), R-24 (spec vs RTL side; hand-off text exists), R-12 (deferred by record — keep or drop) |
| deferred by record | 1 | R-18 (reported, workaround landed and silicon-confirmed; not reproduced with frozen images — closes with the re-triage on the new bitstream) |
| unparsed headers | 21 | a human read of each header, then either the archive section or the open list |

So roughly a third of the 33 close on evidence already recorded, a third are small fixes in this lane, a
third are the RTL lane's or the lead's.

## The board timeline (the only serialised resource)

1. **CHERI volatile program + smoke — NOW** (~45 min of board time, no dependency on anything Capstone).
   Conditions already agreed with the cheri lane: hold the console lock for the whole session; record
   `flash_state.nv_bitstream_name` first; `switch_reset_all`; direct POST with `volatile: true`; bootrom
   banner at 57600 (the console is already at 57600 — the Capstone DTS says so); GDB path (reset halt →
   monitor load_image → set $pc → continue); read `x/7gx 0x80001000` if the UART is silent; power-cycle
   BEFORE releasing the lock; confirm `nv_bitstream_name` unchanged; no Capstone run in between. **LEAD**
   gives the go and answers the SD-slot question. The cheri lane drives it (they own the package); the
   board lane stands by for the console.
2. **Pre-flash boot on the current silicon** (owed to the RTL plan's step 16; batch, control first, one
   unknown last): k800 control → I-4 probes (accum_probe, accum2_probe) → C-3 RV8 rung → the M-1/M-4
   monitor-control domains (see below, only if their commit has landed by then) → the R-25 domain probe
   (INIT rs1≠rd, predicted **1** = duplicate present) → firmware variant D (three CCSRRW `fence.i`
   dropped) LAST — predicted CLEAN on today's silicon by the RTL lane's reading, which makes the post-flash
   D/E boots a no-regression check, written down before the flash. One boot, ~40 min.
3. **LEAD: flash decision** after the RTL lane's synthesis (their Phases 2–4, ~half a day + 3.5 h). Not
   this lane's.
4. **Post-flash boots** (RTL plan steps 18–19): boot 1 = the closing set (k800, six rungs, SLT select1,
   transfer probe) on the new bitstream; boot 2 = R16 acceptance, the S-06 trio, the R20 rung, the R-25
   probe (predicted **0**), variants D and E. Then the `fence.i` drop commit (four sites, R-26 history note
   as the reason, `.c.S` gate), and the registry lines R-25/26/27 → FIXED ON SILICON.
5. **Re-triage boot on the new bitstream**: the six archived-candidate packages as regression tests
   (that is what `ARCHIVED/` is for), R-17/S-01's uc/dp0 pair, R-18's frozen images. One or two boots.

## Work off the board (parallel with 2–5)

### A. Q-06 (bounded investigation, then the fix that follows from it)
Facts: the fault is `sbi.dom`'s `query_region` → `cap_base` on a CPMP-resident region; the value read back
from the slot is untagged; two mechanisms unseparated ((a) installed untagged, (b) read back and not
restored); QEMU's `CCSRRW` is a swap (a read with x0 nulls the entry), the RTL gates the write on the
operand being a capability — UNRESOLVED against the spec.
- Step 1: one more temporary print in the PACKAGE copy of the monitor (assert the `sbi.dom` hash differs
  before running): the tag right after `write_cpmp` in `swap_cpmp` and at the domain-switch boundary.
  This separates (a) from (b). One QEMU run under the lock.
- Step 2: the fix. On QEMU, `query_region` must write back after `read_cpmp` regardless (split_out_cap
  already does); if (a), the install/switch path is the fix and it is bigger (touches the switcher — then
  a monitor commit gated by the FPGA `.c.S` diff and a control boot). Gate: `run-nullblk-all.sh` green,
  the tier, the QEMU probe chain.
- Estimate: the localisation plus the (b)-shaped fix in half a day; longer if it is (a).

### B. The monitor cluster (one or two monitor commits, QEMU-gated, one control boot batched into step 2 or 4)
- M-4: bound `dom_id` in `call_domain_with_cap` (refuse with -1 like the item-1 checks).
- M-1: domains run with `mtvec = 0` — write the trap-vector context slot in `create_domain` (the
  `RTL-domain-trap-vector-unset` folder has the firmware half already confirmed on silicon 2026-09-02;
  check what remains there before writing).
- M-3: SBI ecalls always return error 0 — return the real error where the handler has one; the module's
  failure paths become live, so the module suite must stay green.
- I-5: monitor errors invisible on the FPGA — the "cheap fix identified" in its entry.
- Q-04: QEMU's MOVC must null a NOT_CAP source — an emulator change, gated by the tier (some tests may
  lean on the divergence; the Q-05 experience says find them first with a grep for MOVC-of-scalar shapes).
- Order: M-4 → M-1 → M-3 → I-5 → Q-04. Each its own commit, `.c.S` gate on both targets, pin-bump chain
  only when the FPGA code changes.

### C. Compiler bucket (this lane, after A and B)
- C-37: `lib/Object/ELF.cpp` EM_CAPSTONE relocation names (small; lit test with `llvm-readelf -r`).
- C-9: the redundant `mv rd, rd` around inline-asm constraints (codegen; byte-identity gate on the corpus).
- C-43: a guard for anonymous compiler-generated data under `-capstone-gp-captable` plus a directed test;
  the two known instances are mitigated, the class is not.
- C-3: closes or reopens on the RV8 board reading from step 2.
- C-38, I-02: **LEAD** decisions; C-17 stays latent.

### D. Archiving (docs-only, no board, no build tree; one commit per group) — corrected against the
### folders' own rules (independent inventory 2026-09-09)
The repro README already settles the question: **"A package is open, archived, or resolved-but-retained …
ARCHIVED/ is for packages nothing further will be run against"**, and its 2026-09-07 banner: **"No folder
was moved: sent folders are live links, and the registry is the archive of record."** A sent folder's path IS
the link the hardware side holds, so `git mv` of a sent folder breaks a live report. Therefore:
1. `fpga-repros/`: **no `git mv` of any sent folder.** R16 (bitstream acceptance test), R20 (the R20 rung),
   S06 (the S-06 trio) and S12 are *resolved-but-retained* instruments — they run again on every new
   bitstream (board step 4/5) and stay where they are. Work: a `status` column in the top-level
   `README.md` index (open / resolved-retained / superseded, dated, from each folder's own banner); a dated
   banner where a folder has NO status line (S09, S10, S11; S13's index row says OPEN while its README says
   "cannot be reproduced from this folder" — reconcile from the README); the `ARCHIVED/README.md` index
   row that is missing for `R14-strline-struct/`. Candidates for `ARCHIVED/` only if never sent (check the
   registry's hand-off notes first): R02 (superseded, secondary item) and S08 (bitstream-specific,
   superseded). R01 stays: its banner says "attribution is OPEN".
2. `ISSUES.md`: move the 17 fixed-status entries (18 with R-26 once the bitstream lands — not before) below
   "Archive — fixed, kept for provenance", verbatim (the file's own in-file archive; no second file); read
   the 21 unparsed headers and place each; the six close-on-evidence dispositions written as dated status
   lines. Update the "Last updated" line. One commit.
3. `docs/plans/`: the directory has TWO live rules (`plans/README.md`: "move to ../history/ with a date
   prefix"; `plans/archived/README.md`: "lands here when finished, superseded, or overtaken") — the
   September practice is `plans/archived/` (21 files), so use it and make `plans/README.md` say so (and
   drop its stale row for `18-08-2026_s07-v3-…`, which is already archived). Move, with a "why archived"
   row each: `monitor-unification.md` (Phase A+B complete; fix the two pointers to it in `after-phase-b.md`
   and `current-state.md`), `q03-region-hole-sentinel.md` (implemented and validated),
   `capstone-column-xlang.md`, `xlang-phase1-followups-TODO.md`, `caplifive-system-to-dev-migration.md`
   (its proposal was resolved 2026-09-04). **NOT** `instrumentation-cleanup.md`: it says "do this once S-12
   is closed" and S-12 is closed — it is now an actionable RTL-lane task, not a finished plan (flag it to
   the RTL lane). `after-phase-b.md` stays until its R-26 half closes.
4. `docs/ref/`: `SILICON-BLOCKER.md` stays where it is — both index docs say "do not renumber or trim, its
   line numbers are cited from live repro folders", and a move changes the cited path. `delegation-guidance.md`
   → `docs/history/DD-MM-YYYY_ARCHIVED_delegation-guidance.md` after a grep shows no live reference.
   `docs/README.md` still calls `known-good-controls.md` STALE; the file was refreshed 2026-09-05 — fix the
   line. Nothing in `design/` moves ("decisions, superseded by later ones rather than edited"). `history/`
   untouched.
5. `REPO-MAP.md` / `docs/README.md` counts refreshed in the same commit as 3.

## Session assignment (**LEAD** approves; agreed between the two sessions 2026-09-09)
Two sessions share this lane's checkout: `board` (this one, the original) and `bench` (the fork).
`board` takes A, B, C and every board boot (it holds the board context and the QEMU lock discipline);
`bench` takes D and the `ISSUES.md` re-triage, which touch no build tree and no board, and does not start
until the lead approves this plan. Rules while both run: only `bench` edits `ISSUES.md`, `docs/plans`,
`docs/ref` and `tests/fpga-repros` until it says done (registry lines `board` needs go through it);
`board` owns the monitor/QEMU trees and `tests/runtime-qemu`; `board` commits to `dev` in the main
checkout and messages `bench` before each commit; `bench` works on the worktree branch `archive-docs`
(from bff08f239bd9), rebases onto the new tip, and fast-forwards `dev` at the end; only one session
drives the board; `git commit -o` on own paths, `git show --stat` after each.

## When are we finished
- This lane's QEMU and docs work (A, B, C-37/C-9/C-43, D): about one working week, independent of the
  board.
- The RTL bitstream: the RTL lane's Phase 2 close + 3.5 h synthesis, then the **LEAD** flash decision;
  the two post-flash boots the same day as the flash.
- "Finished" = no open entry in the monitor and compiler buckets; R-25/26/27 FIXED ON SILICON; the old
  R-* entries re-triaged against the flashed bitstream (step 5); the archive moves done. Not in scope:
  R-28 (needs the RTL lane's directed test), R-12 (deferred by record unless the lead reopens it), the
  four LEAD decisions until taken.

## Verification
- Every commit scanned with `precommit-scan.sh` (absolute path), `-o` own paths; the plan itself committed
  to `docs/plans/` after approval.
- Board readings recorded in `tests/board-results/*.tsv` from the run's own transcript segment; a boot
  whose control fails is VOID and rerun.
- Archive moves: `git mv` only (history preserved), banners dated, index rows present, `git ls-files`
  confirms nothing untracked; a `grep -rn "fpga-repros/R01"` sweep updates cross-references.

---
