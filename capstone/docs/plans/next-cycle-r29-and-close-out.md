# Plan 2026-09-10: after the R-25/26/27 bitstream — R-29, the re-triage boot, the SQLite merge, the close-out

*(Board lane. Written at the lead's request after the post-flash cycle closed. The previous plan,
`after-phase-b-combined.md`, is executed except its step 6, which moves here. Decisions that are the lead's
are marked **LEAD** and collected at the end. Nothing here edits CLAUDE.md beyond the sentence the lead
approved on 2026-09-09.)*

## Where things stand

| item | state |
|---|---|
| bitstream | `caplifive_r25r26r27_66c4e7517.bit` on the board (persistent); `caplifive_s12fix_5097eb166.bit` in the console store as the restore path |
| R-25 | fixed on silicon (sw41 → sw45, N=2 in sw47); archived |
| R-26, R-27 | in the bitstream, sim-proven; board arms D/E clean = no-regression; archived; the monitor's four `fence.i` dropped (monitor `1a39e37` … caplifive-system `884b716`, **nested pushes pending the lead's credential** — the parent already points at them) |
| R-29 | NEW, open: a plain `sd` to a granule's HIGH word immediately before a 128-bit `ldc` returns it stale (candidate site `wt_dcache_mem.sv:397`; **audit 2026-09-09 night: PLAUSIBLE-BUT-UNPROVEN** — the store buffer's word-granular disambiguation and the miss-refill leg are live alternatives, a second defect at the same line drives `rd_user_o` from a plain word-0 entry's `.user = 0`, and `r29-lowword` is retracted as confirmation); on every revision including the pre-flash silicon; board sw46/sw48 = 66; sim FAIL 11 adjacent / PASS apart |
| S-06 | struct-assignment acceptance RETRACTED (cited a different program); memcpy half stands; W-12 KEEP |
| registry | 31 headed entries in `ISSUES.md` (29 with a status token; C-4 and the superseded C-14 framing have none), 71 archived |
| SQLite stock-ness | helper's branch `sqlite-stockness` (02a967eb, ae536a64): SQLITE_FEATURE_SET gate + harvests + check-feature-set.sh. **2026-09-09 night, helper:** the eight-define restored image FAULTS at `SQ: E/share1` (cause 24) before the domain enters; bisected one define at a time with an all-deployed control: only `-USQLITE_OMIT_EXPLAIN` breaks it, and EXPLAIN is the one restoration that buys nothing (already inert at the SQL surface). Restore set becomes SEVEN defines; seven-together confirmation, budget rerun and the probe rerun in progress (~1 h). Board boot (`board-b43.sh`) and merge pending on that |
| driver | classifier fix committed (real-segment tested), not yet exercised in a live boot |

## 0. First, the lead (10 minutes)

1. `bash /tmp/capstone/push-final.sh` — the RTL lane pushed `dev` with the gitlink bump on it, so `origin/dev`
   references monitor/wrapper/buildroot/caplifive-system commits that exist only here. Preconditions checked
   (four repos clean, on `capstone-bootstrap`, buildroot pin consistent).
2. Read the **LEAD** list at the end; items 1–3 gate board time.

## 1. R-29 — the fix track (RTL lane; one bitstream; the lead flashes)

The site is located but the mechanism is NOT separated (audit, 2026-09-09 night): three accounts are alive — the
word-granular write-buffer overlay at `:397`, the store buffer's word-granular disambiguation
(`load_unit.sv:297`, `store_buffer.sv:279`), and the miss-refill leg (`:354-358`) — and a second defect at `:397`
(a plain word-0 entry overlays `.user = 0`) needs its own term. So step 0 is now SEPARATION, then the fix.

0. **Separate the accounts in simulation** (RTL lane): one arm per account with instruments that OBSERVE —
   where the `sd` is at the `ldc`'s read cycle (store buffer / write buffer / array), whether the `ldc` hit or
   missed, and the `ldc`'s OWN result register rather than a readback after an `stc`. Predictions first,
   logs kept (the apart-PASS log was overwritten; that result rests on the records file).
1. **Fix candidate** where step 0 points — if at `wt_dcache_mem.sv:397`, extend the granule-scoped term S-10
   added for the tag (`wbuffer_gran_oh`, `:296`) to the data path AND refuse the `.user` overlay for a
   non-capability entry; if in the store buffer, its disambiguation at granule width. **Predictions written
   before the run:** `s06agg-shape` adjacent → PASS with the `ldc`'s own result intact; the 88-row sweep status- and hash-identical, cycle deltas only on
   rows that execute an `ldc` behind a resident plain store, each explained.
2. **Lint = baseline exactly** (UNOPTFLAT 40, ANVIL 0). The file itself says the tag-side term joined a
   combinational ring at 39 → 40 across three formulations (`:384`); `rd_user_o` may be a different cone,
   nobody assumes so. If the count moves, the candidate goes back, not forward.
3. **claim-auditor** on the diff, the soft spot named: "does the fix cover the account step 0 selected AND the
   `.user = 0` overlay, and does it change `rd_ctag_o`'s cone?"
4. **Synthesis** (synth lane, 40 ns, no flow edit), §7 row in the measurements doc. Prediction: within the
   family's spread. **Batching:** any other RTL fix that is sim-verified and lint-clean by then joins the
   same bitstream — the candidates are R-22 (the `stc` arm has never been run; if it is a one-line
   consume) and nothing else; R-28 has no arm that reaches it and stays out. **LEAD** decides the batch.
5. **LEAD: flash decision**, with the lint numbers, the sweep deltas, the §7 row and the step-2 board
   pair reading (below).
6. **Board acceptance** (this lane, one boot): k800 → `s06copy` → the R-29 pair (`s06agg` predicted
   **64**; `s06agg_fence`, the same kernel with a `fence` between the `sd` and the `ldc`, predicted 64
   before and after) → the closing set (six BEEBS rungs, SLT `select1`, transfer probe) → the R-25 probe
   LAST (wedge as predicted). Then R-29 → FIXED ON SILICON, archived; W-12 retired in the same commit.

**RTL lane's own estimate (2026-09-09 night):** half a day to a day of active work to a candidate that reaches
synthesis if the first formulation holds — the granule-scoped comparator exists, but the data path needs the
right entry's DATA and BYTE ENABLES, i.e. a second entry index rather than `wbuffer_hit_idx`; sim pair + sweep
~1 h; lint minutes but the real risk; auditor 20 min; synthesis ~1 h 45 on a quiet machine; if the loop bites,
add a day for reformulation. R-22's `stc` arm is a MEASUREMENT, run in simulation before the bitstream is
defined, never batched blind. The questions the bitstream must answer are written down before it is committed.
The audit landed the same night: PLAUSIBLE-BUT-UNPROVEN, so step 0 precedes the candidate.
The board boot after the flash: 1 h.

## 2. The re-triage boot on the current bitstream (this lane, one boot, ~1 h, no fix needed)

Batched, control first, one expected-to-wedge domain last. Every arm here has a written prediction; a boot
whose control fails is VOID.

| position | arm | question | prediction |
|---|---|---|---|
| 1 | `k800` | control | 4 |
| 2 | `s06agg_fence` (new: the S-06 kernel with `fence` between the `sd` and the `ldc`) | is the board pair discriminating BEFORE the fix | **64** (`s06agg` read 66 twice on this bitstream; not rerun) |
| 3 | `s06agg_lowword` (new: the last plain store to the LOW word, high word stored early) | the `.user = 0` overlay account on silicon — the audit says a RESIDENT word-0 plain entry should zero the high half, so 66 here would be that defect; the sim twin `r29-lowword` is retracted (its store and load share `[11:3]`, which stalls the load) and the board twin carries the same confound | **64 or 66; either reading is recorded, neither attributes** |
| 4 | R-15's 9216-byte capability-bearing global | attribution retracted 2026-07-31, never rerun on a fixed bitstream | returns (value from its host oracle) |
| 5–6 | R-17/S-01's `uc`/`dp0` perturbation pair | the pair was never run in the 2026-09-05 sweep | both return; if one hangs, it hangs LAST |

R-17's pair goes last because one arm may hang. The R-25 probe is not repeated (N=2 already). **R-18 is NOT a
board arm here** (RTL lane, 2026-09-09 night): read in granules, its `rmB`/`rmC` pair is exactly "the RMW scalar
shares the victim's granule (offset 4, low word; victim at offset 12, high word) — damaged" versus "a different
granule — correct", which fits granule-scoped forwarding sharply and makes the capability store two rows away
the red herring. But `r29-lowword` (an adjacent plain store to the LOW word) PASSED, which the simplest form of
that account would not predict (that arm is since RETRACTED as a measurement, so it constrains nothing either
way); the difference may be RMW vs plain store, or the victim being reached through the movc-zero copy path.
The RTL lane's first four R-18 arms (the victim's own store drained by a fence: both LDC-read arms fail, both
plain-read arms pass, same-granule and other-granule alike) do NOT reproduce the rmB/rmC discrimination and are
confounded by the same stc-then-readback path the audit flagged — not yet a result. So R-18 gets a DIRECTED SIMULATION ARM of its exact geometry (RTL lane, after the
audit), not a board boot — the board could not separate the two mechanisms either, and "not reproduced with
its frozen images at `5097eb166`" was never a mechanism verdict. If the arm attributes R-18 to `:397`, R-18's
question joins the list the R-29 bitstream must answer, and its frozen images go into the acceptance boot. The
classifier fix gets its live positive check here for free if any arm wedges; otherwise it stands on the
real-segment test. Records: tsv rows, the R-18/R-15/R-17 registry lines, the R-29 entry's board section.

## 3. SQLite stock-ness (helper delivers; this lane audits, boots, merges)

1. Helper: confirm the seven-define restore set composes (running), rebuild and re-budget the seven-define
   image (point 3), commit the revised restore set and the domain feature probe (deployed arm reads 0/6 with
   marker 0x4EB00000; the probe must call `sqlite3_config(HEAP)` + `sqlite3_initialize()` itself because the
   build sets OMIT_AUTOINIT — without them the domain faulted as cause 1 at pc 0 = M-1 reporting someone
   else's fault), rerun the probe on the restored image (expect 7/7 vs 0/7). About an hour.
2. This lane: the audit's point 3 against the primary sources (the six gated C APIs in
   `check-feature-set.sh`, the SQLITE_RESTORE literal appended after SQLITE_DEFINES, both harvests);
   negative test: `check-feature-set.sh` must FAIL on the deployed image and PASS on the restored one.
3. Board boot `board-b43.sh` on the SEVEN-define image (k800, restored domain on `select1.test`,
   `negative-control.test`, `slt_lang_aggfunc.test`, then the feature probe), predictions = the branch's QEMU readings (select1 1031 records completed,
   the negative control's tally, the re-baselined aggfunc tally). Separate boot from section 2: SLT
   stages are long and one SLT domain per boot is the rule.
4. Fast-forward `dev` from the worktree branch after review; `git show --stat` after.

Estimate: half a day after the helper delivers.

## 4. The 29 open registry entries — disposition and owner

| bucket | entries | disposition |
|---|---|---|
| **RTL lane, fix or one sim arm** | R-29 (section 1); R-22 (`stc` consume arm, never run: one 14-s test, then fix or spec note); R-21 (sim-confirmed spec violation; with the lead as board owner "NOT yet reported" is moot — the token drops, the fix joins a bitstream when written); R-13 (a duplicate of R-21's `cincoffset` half — fold into R-21, ID retired); R-19 (mechanism unconfirmed; one arm on the new bitstream if its image is frozen, else parked); R-24 (spec direction determinate — **LEAD** rules spec vs RTL, then a one-line RTL fix or a spec erratum); R-28 (no reachable arm; stays open, no work) | 7 |
| **board re-triage (section 2)** | R-15, R-17/S-01; R-18 by a directed sim arm (RTL lane) first | 3 |
| **this lane, monitor/QEMU, board-free** | M-1 (`ctvec` slot in `create_domain`: the firmware half; the RTL half — a domain fault wedges — is the M-1 RTL entry, joins a bitstream); M-5 (mint the re-share `UNINIT` with CAPTYPE instead of a revoke-derived one — the R-25 lesson; QEMU gate + board control); Q-07 (QEMU `INIT` accepts `cursor > end`; tier gate); Q-04 (QEMU MOVC nulling — **LEAD + RTL** spec ruling first); I-03 (latent, a linker check; one commit) | 5 |
| **compiler lane** | C-43 (mitigated in-branch 5d2932a9 — land on dev, then archive); C-14 (root-caused: the `movc` scalar copy — status to FIXED or a one-line fix); C-4 (header has no status token — the remaining domain-creation half needs a human read); C-17 (latent, stays) | 4 |
| **close on evidence — LEAD confirms** | R-4, R-5 (no reproducer, none produced by the sweep: CLOSED, not reproducible); R-10 (the capstone-c declarator fix landed 2026-07-29 and is in the shipped firmware; its secondary `is_cap_req` note becomes a one-line RTL observation — FIXED); R-3 (worked around by the distinct-VA rule enforced by preflight C15 — WORKED AROUND stays, or CLOSED as a design constraint); R-11 (`check-repr.py` on the current corpus, no boot: closes if no hit); S-04 (the blamed form did not reproduce at N=1 on `5097eb166`; **LEAD** moves the silicon default or the entry stays); S-10/S-10b (in the bitstream by content; S-10b's fix unsynthesizable — CHARACTERISED, kept) | 7 |
| **LEAD decisions** | C-5 (4 KiB code window: design limit or defect), C-38 (mnemonic), I-02 (ID allocation convention), R-12 (rev-node exhaustion silent: keep deferred or schedule) | 4 |

Registry mechanics as established: header token decides the file; a final status moves the entry to
`ISSUES-ARCHIVE.md` in the same commit; IDs never reused; one folder per issue.

## 5. The S-06 folder now carries two issues (**LEAD**)

`S06-untagged-ldc-stc-high-half/` is a sent link and now holds R-29's reproducer, sim arms and records
beside S-06's — the shape CLAUDE.md says has had to be split after a link was out. Recommendation: create
`R29-wbuffer-highword-forwarding/` now (copy `src/s06agg*`, `sim/s06agg-shape*`, `sim/r29-lowword*`, the
board rows, the disassembly, the mechanism at file:line; `SHA256SUMS`; index row), leave S-06's folder
with a one-paragraph pointer, and add the R-29 board pair from section 2 to it. One hour, board-free.
The alternative — split at fix time — leaves the live S-06 page showing R-29's evidence for as long as the
fix takes.

## 6. Order and timeline

| step | who | board | blocks on |
|---|---|---|---|
| 0 push-final.sh | lead | — | nothing |
| 5 R-29 folder | this lane | — | LEAD yes/no |
| 2 re-triage boot | this lane | 1 boot | nothing (images: two new S-06 variants + R-15's and R-17's from the attic) |
| 3 SQLite boot + merge | this lane | 1 boot | the helper's delivery |
| 4 monitor/QEMU items (M-1 fw half, M-5, Q-07, I-03) | this lane | control boot batched into 1.6 | nothing |
| 1 R-29 fix → synthesis | RTL lane, synth lane | — | nothing; flash is LEAD |
| 1.6 acceptance boot | this lane | 1 boot | the flash |
| 4 compiler items | compiler lane | — | nothing |
| 4 LEAD items | lead | — | — |

Day 1 (2026-09-10): 0, 5, 2, the helper's delivery and 3; the RTL lane starts 1 and the R-18 arm. Day 2: 4's
monitor items, R-22/R-21/R-13 sim arms, the R-29 candidate through lint and audit, synthesis overnight (a
day later if the lint ring bites). Day 3: LEAD flash
decision, the acceptance boot, R-29 archived, W-12 retired. "Finished" = every open entry either fixed,
closed on evidence, or explicitly parked with its blocker named (R-28, R-12 by decision, C-17 latent).

## Verification

- Every board arm has its prediction in this file before the boot; readings recorded from the run's own
  transcript segment into `board-results/*.tsv` with the image hash; a result cited by hash, never by label.
- R-29 fix: the separation arms + the sim pair + sweep identity + lint baseline + auditor + synthesis row, in
  the commit message; the board acceptance is the rung's 66 → 64 with the pair unchanged.
- Every commit scanned with `precommit-scan.sh` (absolute path), `-o` own paths, `git show --stat` after;
  nested pushes through `push-final.sh`.

## LEAD decisions, in order

1. Run `push-final.sh` (section 0).
2. R-29 folder now or at fix time (section 5). Recommended: now.
3. Which RTL fixes batch into the R-29 bitstream (R-22 if ready); then the flash decision when synthesis
   is done (section 1.4–1.5).
4. R-24 spec vs RTL; Q-04 spec ruling (with the RTL lane).
5. Close on evidence: R-4, R-5, R-10, R-3, S-04's silicon default (section 4).
6. C-5, C-38, I-02, R-12.
