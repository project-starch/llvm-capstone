# Current Capstone state

Minimal snapshot. Read first in every session.

## 2026-09-04 — CURRENT

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
* **SQLite RUNS ON SILICON — that is a LIVENESS result, not a correctness one.** The `slt/`
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
