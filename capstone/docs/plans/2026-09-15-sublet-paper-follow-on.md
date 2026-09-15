# Plan: the Sublet-paper follow-on on the board (2026-09-15, morning)

## Context

The 2026-09-14 plan is executed: E1 (the S1/S2 matrix, 21 cells × 3 repetitions, every cell identical),
E2 (P1's matched pair at size 1: ⑥/⑤ = 1.1024 at -O0), E3 (R1, 420 records at five repetitions: REVOKE
23.0 cycles per node, the fill 1.81 cycles per byte, flat in heap and depth), E4 (H1's calibration: a
dependent load 9.0 cycles in the data cache, 48.2 from DRAM; the timer 2 cycles) and E5 (M3's ledger)
are on silicon, bundled in the paper's record shape on the local worktree branch
`board/e1-s1s2-hardware` (`~/capstone-artifacts/paper-wt`, 96684c8), and reported in
`docs/ref/fpga-silicon-measurements-for-paper.md` §7r–§7v. On the way, the -O1/-O2 Sublet divergence was
root-caused to C-32 live on silicon (the RTL's `movc` nulls an untagged source and the port's
`setupLookaside` re-reads it), masked by Q-04 on every emulator pass and confirmed by arm C; the -O2
compound stands on read configurations (⑤-O2/native 1.3624, ⑥(arm C)/⑤ 1.1797, 1.607 full stack); the
preflight's false refusals were measured (SIGPIPE under `pipefail`) and fixed; and the archive audit
found split transcript lines common but no verdict or cited number affected (M-11).

**What this plan does.** The lead's four decisions of this morning set it: **C-32 is fixed by design A**
(the rematerialisable bridge pseudo), **M2 runs now as a bounded diagnostic**, **the board branch is
pushed to the paper repository**, and **R-21/R-22 get one boot** for their silicon reading. Around those:
the M-11 structural fix (the drivers' summaries derived from a seam-joined parse, with a positive
control), the bundles' hygiene before the push, and the desk preparation for M1's day.

**Standing constraints that shape every item.** One board, serialized, ask-first reflash; one Sublet
SQLite workload and one 128 MiB arena workload per boot; ≤ 12 region-bearing harness invocations per
boot (M-9) and ≤ 80 % of the 65,532 nodes cumulatively per boot (METHODS:80); a capability fault inside
a domain wedges (M-1), at most one expected-fault arm per boot, last; results cited by image hash from
the run-scoped transcript, seams joined before any line-based read (M-11); five measured runs per board
point over ≥ 3 boots, fresh domain each, no warm-up in the timed bracket (METHODS:92–95); an emulator
pass of an optimised image cannot stand in for the board on any path that casts integers to pointers
until Q-04 and C-32 are resolved (§7s). Never edit the paper's prose; `studies.json` and the
manuscript are the paper lane's; `capstone/paper` is never pushed by this lane.

**Three things this plan surfaces for the lead rather than decides.** (1) The S1S2 bundle carries six
`unsafe-success` cells (s3/s5 ×3: a stale load retires through a tagged pointer, the M-mode gate), and
METHODS:86 says "an enforcement or oracle failure stops dependent performance publication" — whether
that binds P1/R1 timing publication on this configuration, or the gate is a labelled configuration
fact, is a framing call. (2) Q-04 (whether a scalar `movc` source must be consumed) stays the open spec
question; design A removes the compiler's exposure without ruling it. (3) M1's reclaiming node table
(the RTL lane's design doc, awaiting the audit order) gates complete P1, R2, M2's active-nodes series,
M4 and M5; nothing here substitutes for it.

## Work items, in the order they run

### F1 — C-32 by design A: the compiler lane's fix, then one confirming boot

*Compiler lane (hand-off, with the lead's choice attached).* Replace the bare
`DAG.getTargetInsertSubreg(Capstone::sub_cap_addr, …, getUNDEF(c128), Addr)` at
`llvm/lib/Target/Capstone/CapstoneISelLowering.cpp:8091-8094` (the `inttoptr` lowering) with a pseudo
`(outs GPCR:$rd), (ins GPR:$rs)` marked `isReMaterializable`/`isAsCheapAsAMove`, expanded to `ADDI` on
`$rd`'s `sub_cap_addr` — the same `ADDI` `copyPhysReg`'s GPR→GPCR arm already emits
(`CapstoneInstrInfo.cpp:679-690`). Notes the exploration turned up for them: `PseudoTRUNC_CAP`
(`CapstoneInstrInfo.td:2569-2591`) is the precedent but its `PseudoInstExpansion` maps 1:1 and cannot
write a sub-register, so the expansion belongs in `CapstonePostRAExpandPseudo`; the remat hook
`isReallyTriviallyReMaterializable` special-cases only RVV; `MachineCopyPropagation` in `addPreEmitPass`
deletes `addi rd, rs, 0` when `rd == rs` (the reason `PseudoTRUNC_CAP` stays a pseudo). Do NOT touch
`MOVC`'s definition (C-46's box: tying `$rd = $rs1` breaks the copy primitive). The test
`llvm/test/CodeGen/Capstone/c32-movc-untagged-live.ll` needs its `-O0` RUN line given its own prefix
(its `real_cap_copy` control has no `movc` at -O0 today, so it would fail after the fix as before), the
`XFAIL` removed, and a PHI-shaped function added as the known residue. Gate for "fixed":
`capstone/tests/movc-cfg-scan.py` over the rebuilt -O2 Sublet image reads 0 integer-only sites (the
mixed and call-return buckets reported beside it), lit green, the QEMU suites green.

*Gate corrected 2026-09-15 14:15, from the compiler lane's result (46c53b7b6ae2, 19bc05cf21b1 on their branch; scan 1/0
against 1/1 before; pair at `~/capstone-artifacts/c32-fix-2026-09-15/`, before `ec061577fb008e18`, after `a1f8f2093696d511`):*
design A's accepted residue is itself an integer-only site — `renameResolveTrigger`'s block-entry copy live around its
back-edge, pinned by `bridged_phi_residue` in the test — so "0 integer-only sites" cannot be met and would invite a
weakening. The gate is **0 integer-only sites other than that PHI residue, enumerated by function and offset**, with the
mixed and opaque buckets reported beside. And the emulator pass for the image is SQLLogicTest on the -O2 SQLite image with
the `movc` density read back (17,378 vs ~6,755 at -O0), not the standard suites: `ptr_int_ptr_roundtrip.c` forms no
`PseudoBRIDGE_CAP` (its `volatile` pushes the value through memory), so the suites do not exercise the lowering. The fix is
not on dev: the scan's range mode blocks the merge on author lines of the collaborator's S2 commits already on dev, and
whether authorship metadata is in the no-names rule's scope is the lead's ruling. The boot waits for dev.

*Board lane, after the fix lands on dev and the toolchain is rebuilt (never during a suite).*
Rebuild cell ⑥ at pure -O2 (`build-sqlite-silicon.sh`, `SQLITE_OPT_LEVEL=-O2`, no `SQLITE_OPTNONE_FUNCS`),
scan it, emulator run at the default and the 2 MiB arena (the counters must be
5568/37966/32565/37966/5401 at 2 MiB), then ONE boot with `board-c6var.sh`: control → ⑥-O2 pure at
2 MiB → control → probe. Pre-registered: counters equal to the emulator's; cycles within 0.1 % of arm C's
1,376,190,813 (arm C differs only by one setup function at -O0, off the timed path). Then ⑥-O2/⑤-O2
(1,166,594,074) is P1's -O2 `protection_cost` at size 1, labelled bounded-prototype for size and for M1,
with no workaround in the label. §7s gains the reading; C-32 → FIXED with the scan as its gate.

### F2 — M-11 made structural: the drivers' summaries derive from one seam-joined parse (desk)

One module, `capstone/tests/rtl-smoke/fpga_driver/transcript.py` (stdlib only): `read(path)`
(utf-8, errors replaced), `scope_to_run(framed)` (after the last `monitor load_image`, on the FRAMED
log), `uart_chunks(framed)` (every `[fpga] [uart] <repr>` line through `ast.literal_eval` — both quote
styles, the `+0B` form; an unparseable frame is an error unless it is the last line of a live log),
`uart_text`, `strip_markers` (the unanchored `[A-Z0-9]{4}:[0-9A-F]{8}\n?`, optional `SQ: ` lines),
`last_marker(joined)` (the last `SHA[56]:` on the UNSTRIPPED joined text), `arm_segments(framed)`
(split on `[stages] --> TEST`, each arm carrying its framed slice, its joined UART, its end line and
`returned`), `find_all` (full matches), `require` (empty input is an error), and a CLI
(`uart`, `last-marker`) for the watchdog and for reading a transcript by hand. Two-text discipline
in the docstring: marker rows and the stall marker read the joined text; domain lines read the
stripped text.

Then, in this order, each with a replay over the archive as its check:
1. `test_transcript.py` — the positive control: a synthetic framed log with the control line split
   across two chunks and an `[event]` line between them, a double-quoted chunk, a marker mid-token
   inside `SPEEDTEST1-CYCLES 1166594074`, a `SHA6:` split across chunks, and `ngx retval = 1309490692`
   split after four digits; every test asserts the OLD reading is wrong and the new one right.
2. The watchdog (`board-watchdog.sh:131`): `last-marker` through the CLI, the per-line `sed` as the
   fallback; board-b76 must still read `SHA5:00000001`.
3. `board-b78-w2h.sh`: the per-arm table from `arm_segments`, the mark regex anchored on its line
   terminator so a cut number can never print as a mark; replay over the 15 E1 directories, the only
   delta r3b3's subpool cell.
4. `board-c6var.sh`, `board-b80s.sh`, `board-b80a.sh`: same pattern tuples over the joined/stripped
   texts, values anchored on the next token, banners counted on the joined text; replay expected
   deltas: optC2's `sublet:` 0x→1x, b80s-O2's 8,562 → 1,166,594,074 in band, control rows 1x→2x.
5. `board-r1.sh`, `board-r1e4.sh`: read driver.log (boot.txt has no banner: the "must be 1" line was a
   structural zero), `r1-lines.txt` byte-identical for r1-b2…b8.
6. The marker tail in all six: `refused` when rc≠0 and the log says `preflight: BLOCKED`, `failed`
   on any other rc≠0 or a failed summary, `done` only on rc=0 — and a non-zero exit; the session
   monitors accept the third value.
7. `ISSUES.md` M-11 → GATED (the module and its test named; the two new findings: boot.txt carries no
   banner, ENTRY-STALL lives in watchdog.log). Follow-up: `r1-bundle.py` and `e1-bundle.py` import the
   module (both drop double-quoted chunks today).

### F3 — Bundle hygiene, then the push of the board branch

On `board/e1-s1s2-hardware` in `~/capstone-artifacts/paper-wt`: add the `.gitignore` negation
`!experiments/results/**/*.log` (METHODS:154 wants raw output kept; the paper's `*.log` rule silently
drops every `*-driver.log`/`*-watchdog.log`, 30 of 60 S1S2 raw files) and stage the untracked raws;
delete the leftover `experiments/results/S1S2/sw78-rep1/`; populate `analysis/` for S1S2 and R1 (the
summary tables as CSV plus the slope/fit script that produced 23.0 cycles per node and 1.81 cycles per
byte — versioned, so a reader can rerun them); fix H1's `clock_caches_memory.memory_latency_calibration`
(null against the filled calibration block). `make experiments-check` and the scan from inside the
worktree before each commit. Add the branch to `~/.claude-c/secrets/push-allowlist.txt` (the lead's
instruction of today), push it, and hand the paper lane the evidence-state mapping for `studies.json`
(theirs to edit): R1 pending → partial (capacity-bounded, five repetitions, no M1), P1 pending → partial
(bounded-prototype diagnostic at size 1, -O0 and -O2 with the C-32 label until F1), S1/S2 partial with
the FPGA matrix ×3 and the PostgreSQL cells unsupported (M-8), H1 partial (calibration and instruction
tests filled; per-module area and one family pending), M3 partial.

### F4 — R-21 and R-22 on silicon: one boot, then the two handover folders

A `--series linear` in `capstone/sublet/r1/r1_slots_pools.c`, plain-alias fixture, every arm returning a
mark (no expected fault): (a) a LINEAR capability in a register, `cincoffset rd, rs` with `rd ≠ rs`,
then the source's type read back (`sublet_type` on the stored source: spec says cleared → 7; R-21 says
still LINEAR → 0); (b) the same for `scc`, `tighten`, `shrinkto`; (c) `init` on an UNINIT region with
`rd ≠ rs`, both registers read back (R-21's duplicate); (d) `stc` of a LINEAR register, then that
register's type (R-22: not cleared); (e) the NONLIN control for (a) and (d), which must NOT clear
(the scan's `must_stay_silent` shape, `tests/rtl-smoke/linear-clear-controls/README.md`). Emulator pass
first (the emulator's own readings recorded per arm; Q-list entries if it differs from the spec too),
then one boot batched as one invocation list after a control, three repetitions in one boot (the arms
return in microseconds). Pre-registered from the simulation repro (`linear-clear-audit.S`, 12 s): every
R-21/R-22 arm reads "not cleared" on silicon, the NONLIN controls unchanged. Then, per the handover
rule: `capstone/tests/fpga-repros/R21-linear-source-not-cleared/` and `R22-stc-source-not-cleared/`,
one issue per folder, each README self-contained with the hash-cited board reading, the simulation
repro and the spec lines; `ISSUES.md` R-21/R-22 → "confirmed on silicon, reported"; H1's
forbidden-linear-copy family → measured (as the failing test it is, with the registry reference).

**Outcome (12:53):** every probe read the SPECIFICATION on silicon — `cincoffset`/`scc` clear the linear source,
`ldc` empties the slot, `stc` nulls the register — with both controls behaving; R-21 is partly resolved and R-22
resolved on the deployed bitstream, no handover folders are needed, and the emulator is the divergent side on
`ldc`/`stc` (Q-12). §7w.

### F5 — M2's bounded diagnostic: working-set and data-only series (about six boots, unattended)

`--series chase` in the R1 harness, from E4's chase: N 64-byte records, each holding the INDEX of the
next (never a forged capability), a random single cycle built by Sattolo's shuffle from seeds 1/2/3, a
separately counted lookup array of valid capabilities, two warm-up traversals then 100,000 dependent
accesses inside the bracket, checksum and visit count verified, the cold first traversal reported
apart (`M2-access-path.md:17-20, 41-44`). Arms: **custom-spatial** = the records in one plain alias of
a carved region; **custom-sublet** = each record its own Sublet object (a leaf alias per record under
one handle, the port's own discipline), the same addresses, bounds and access count; **data-only** =
the spatial chase and data sizes with the lookup array but no temporal query (the labelled
non-protecting ablation). Points: working-set 16 / 64 / 256 / 1024 / 4096 live records (1 KiB–256 KiB,
spanning E4's calibrated 9.0 → 48.2 knee); the active-nodes series is DEFERRED to M1 (`M2:12`, repeated
turnover). Node budget per boot: a 4096-record sublet point mints 8,192 nodes, so at most six sublet
invocations per boot beside spatial ones; schedule 3 seeds × 5 runs per point as fresh-domain
invocations over ≥ 3 boots (METHODS:92), ~45 invocations, ~6 boots, the invocation list in a script
(M-10), `--tables` above every arena (M-9). Report per run: raw and calibrated cycles per access,
`minstret`, the node count, working-set bytes, and E4's latency calibration beside it. **The claim line, written
into the harness, the bundle and §7 (corrected by the RTL lane's reading):** each access reads the next
INDEX with a plain load, which carries no temporal query on this bitstream (the LSU's check for plain
loads through a capability base is M-mode-gated — E1 measured stale loads retiring in a domain), and
then fetches the next record's capability from the lookup array with an LDC, which DOES run the DYN
unit's node-validity query at every privilege level — so the series measures that query under a
growing set of live nodes (one alias per record in the sublet arm against one per region in the
spatial arm), which is the study's question; stale-data ENFORCEMENT on the plain load is what it does
not measure. No node-cache knee or node-size inference is made (`M2:57-58, 63-65`). Enabling the LSU
check below M-mode is an RTL edit (one hardcoded privilege literal), a synthesis run and an ask-first
reflash, not a knob, and the RTL lane notes capmode and M-mode may be mutually exclusive there — to
confirm before anyone designs around it. **Pre-registered shape and two controls (the RTL lane's reading of the node unit):** the
validity query is ONE indexed 16-byte read (`get_rev_node`, no walk — REVOKE walks, the query does not), so
its cost is flat in the number of live nodes algorithmically, and what grows is the node table's footprint
against the 32 KiB data cache (a node is 16 bytes = one 128-bit line, so 2,048 nodes if nothing competes).
The quantity is TOUCHED nodes — one per access, the node of the capability the LDC dereferences, so one per
record in the chase — not minted nodes (a record's carve mints about two, and the size-1 SQLite run mints
~8 per hand-out cycle); the harness prints both, and the pre-registration names touched. Pre-register
flat-then-inflect, with the inflection near two thousand TOUCHED nodes; keep the aliases' ids sequential
(monotonic head allocation) so an eight-way conflict pattern does not put the knee earlier; the alternative (a slope from the first point) is distinguishable
in one series. Controls: (a) a spatial arm carrying the SAME number of live nodes as the sublet arm (leaves
minted and held, not used), so both arms share the node-table footprint and the per-access difference is
isolated from the footprint effect; (b) points below two thousand live nodes so the flat region is visible
(16 / 64 / 256 / 1024 records are below it, 4096 above). The finding to write, if it reads so: the temporal
check costs one indexed read, and the cost of a growing alias set is a locality cost in the node table,
not a mechanism cost. Bundle at `experiments/results/M2/fpga-<date>/` in the record shape.

**Correction (2026-09-15 14:00, written after two of the four boots had run and before any number was read as a
finding):** the pre-registration above mis-specifies the quantity. The timed access is `cur = *lookup[cur]` — an LDC
whose ADDRESS capability is the lookup array's capability, then a plain load through the loaded record capability —
and on this RTL the DYN unit's node-validity query is on the LDC's address capability only (Q-11: a loaded
capability's node is not queried on the load; the plain load's LSU check is gated and its exceptions are lost, R-34).
So the access path queries ONE node per access in both arms; the harness's `touched` field is a label, not a count of
queried nodes. What the pair measures is whether the per-access query cost depends on the number of LIVE nodes (the
sublet arm mints about 2N+4, the spatial arm 6). The footprint of a growing QUERIED set is not measured by this design;
it would need the record to hold the next capability, which the study's design excludes. The bundle carries the original
pre-registration verbatim followed by this correction; the §7 entry reads the pair under the corrected scope.

### F6 — Prepared for M1's day (desk, no board)

The four-arm M1 driver (drop / retain-ring / retain-pressure / release-retained to 10 C cumulative node
allocations, snapshots every C/16, isolated stale probes at each checkpoint, `M1:30-35, 53, 63-65`) as a
`board-m1.sh` over the R1 harness's fixture, runnable the day a reclaiming bitstream is flashed; the
retain-pressure arm is sw74b's exhaustion diagnostic already on record. No boot until then.

**Prepared (2026-09-15 14:25):** `--series m1` in the R1 harness (`--arm drop|ring|pressure|release`, `--cap C`,
`--stale-take` for the faulting probe, last): sixteen live 64-byte objects replaced round-robin, one release (revoke +
init) and one take (one node) per allocation, to 10C; the ring keeps the 16 most recent obsolete aliases in tagged
memory, the pressure arm every one in a preallocated 2,048-entry buffer (a full buffer stops as `buffer`, not as
exhaustion), release clears them and runs 2C more; a snapshot every C/16 prints minted/revoked/init/retained and
non-faulting type reads of the oldest retained alias and a live slot; the budget stop is the deployed table's
exhaustion classification. Emulator flow check at C = 64 (build10 `848887ae81b8c0e3`, host `2c9e82d101b48160`): all
four arms to target, minted = 31 + one per allocation, revoked = allocations, retained 0/16/640/0 (release 768
allocations), the retained alias reads type 7 on the emulator (Q-11; 2 expected on silicon). Driver: the parametrised
R1 driver with `f6/invocations-m1-*.txt` and `chain-m1.sh` (boot 1 = the four arms at the 256-entry diagnostic
capacity, boot 2 = the pressure arm alone to the 80 % budget on the deployed table) — prepared, not launched: the
no-reclamation baseline is the study's other configuration and whether to spend two boots on it now is the paper
lane's and the lead's call; the four arms at C = 65,532 need the reclaiming bitstream (one arm per boot). What the
harness cannot read and a reclaiming build must expose: occupancy and free-node counters, the reclamation count, and
a node id/generation read for the identifier-turnover witness. Subordinate handles retained across a release are not
modelled yet.

*15:05, after the paper lane's review:* every snapshot now also prints the interval's raw cycles in the takes (`take_cyc`,
minting: `mrev`) and in the gives (`give_cyc`, release: revoke + fill + init) with the interval's allocation count, so
the two cost curves against cumulative allocations come out of the same boots (build11 `9a01b12a4db639b6`); on the
deployed table the retention pattern varies nothing (no reclamation: one node per allocation whatever is retained), so
if the two boots go they are labelled **no-reclamation baseline**, never M1, and their headline is the minting and
release cost as the table fills — pre-registered flat for both — with the bound as corroboration and the pattern
comparison void until a reclaimer exists. *Reconciled with R1 (15:20, after the paper lane's check):* the shape is R1's
`individual` one — a carved LINEAR leaf handed out as a delin'd NONLIN alias under a REV handle in its slot — so the
release has NO fill (the alias's revoke returns the region LIN; R1 measured `fb = 0`), and the earlier "plus a 64-byte
fill" was wrong. Brackets by primitive: `take_cyc` = LDC + MREV + STC + DELIN; `give_cyc` = LDC handle + REVOKE + LCC
type + STC ×2. Predicted from R1's 64-byte object medians (revoke 51, type 12, init 2, reissue 355 − the spatial arm's
36): PRIMARY, what the boots test — `take_cyc/n + give_cyc/n` FLAT in cumulative allocations (least-squares slope
times the run's allocation range below 1 % of the mean; last quarter within 1 % of the first); SECONDARY, scored
apart — the sum in the band **384–391** raw cycles per allocation (384 from R1's phase medians; 391 = R1's total 593
minus bookkeeping 166 minus the checked first use 36, the seven between them being R1's six timer reads against
these brackets' four), where outside the band refutes the magnitude model and leaves the primary result untouched.
The split between the two brackets is reported, not predicted. Boot 1's curve at the 256-entry capacity is labelled diagnostic, not a hardware result — by
capacity, not by platform (`M1-node-reclamation.md`).

## Hand-offs (cross-session messages, roles only)

* **Compiler lane:** F1's design A with the lead's choice, the code sites, the test's `-O0` arm, the
  PHI-residue case, and the scan as the gate; Q-04's emulator side is untouched by it.
* **Paper lane:** the `studies.json` evidence-state mapping (F3); the METHODS:86 question on the
  `unsafe-success` cells for the lead; the `.gitignore` negation and the pushed branch; M2's caveat
  before its numbers are taken; the E4 pair (loads 5.36× residency-sensitive, the fill 1.03×) already
  in §7v.
* **RTL lane:** after F4, the two folders (one link each, no message body); the M1 reclaimer design's
  audit order is the lead's; M2's interpretation asks whether the LSU node check is to be enabled in
  domains on a future bitstream (ask-first reflash).
* **Synth lane:** H1's per-module area extraction (single seed; `run.tcl` has no seed hook, a flow
  change for the lead).

## Order and board time

F2 first (desk; every later boot is read through it), F3 (desk, then the push), F4 (one boot, ~15 min),
F5 (~6 boots, unattended, overnight), F1's boot when the compiler fix lands. Every boot: control first,
pre-registration in the driver header, image hash in the record, the seam-joined transcript as the
record.

## Verification

* F1: scan reads 0 integer-only sites on the pure -O2 image; the board's counters equal the emulator's
  to the unit; cycles within 0.1 % of arm C.
* F2: `test_transcript.py` passes and each test's old-reading assertion holds; archive replay: 0
  unparseable frames over the 86 logs, the module never reads fewer than the seam-join, the five
  control boots flip 0→1, `r1-lines.txt` byte-identical for r1-b2…b8; board-b76 still reads a stall.
* F3: `make experiments-check` passes; every raw file tracked; the push lands on the paper remote.
* F4: the NONLIN controls read unchanged (the positive half); each R-21/R-22 arm's board reading
  matches the simulation repro; two folders, one issue each, hash-cited.
* F5: checksum and visit counts equal on both arms at every point; the spatial curve reproduces E4's
  9.0 → 48.2 shape; five runs per point over three boots; node demand ≤ 80 % per boot in the manifest.

## Documents to update (with each item)

* `docs/ref/fpga-silicon-measurements-for-paper.md`: a §7 entry per boot (F1, F4, F5) and the M-11
  closure note (F2).
* `docs/ref/ISSUES.md`: C-32 FIXED with its gate (F1); M-11 GATED (F2); R-21/R-22 confirmed on silicon
  and reported (F4).
* `docs/state/current-next-step.md` / `current-state.md`: the header per item; the lead's three open
  questions listed as such.
* The paper repository (board branch): the bundles under `experiments/results/`, `analysis/` populated,
  the `.gitignore` negation; no manuscript edit from this lane.
* This plan lands as `docs/plans/2026-09-15-sublet-paper-follow-on.md`.
