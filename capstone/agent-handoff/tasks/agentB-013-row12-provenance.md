# Agent-B task 013 — resolve the row-12 provenance citation-mismatch (expo-sqlite unfinalized-close)

*Hand this whole file to Agent-B (`claude-b`), clone `/home/alexey/dev/llvm-capstone-b`,
branch `capstone-bootstrap-b`.*

---

You are **Agent-B** (compiler/codegen + emulator + literal-repro lane). Obey
`./CLAUDE.md` and `capstone/agent-handoff/{MULTI-AGENT-WORKFLOW,COORDINATION}.md`.

## Why this task

A just consolidated the 19-row corpus into a single provenance ledger
(`capstone/benchmarks/sqlite/cve-repros/PROVENANCE-LEDGER.md`, on `capstone-bootstrap`).
It surfaced **one hard citation-mismatch that is a real paper liability**: the
paper's `tab:scope` caption claims "All links verified," but **row 12** cites an
artifact that documents a *different* defect than the row describes.

- **Row 12** (`row12_expo_unfinalized_close`) essence: a binding *ignores
  `SQLITE_BUSY` on close, clears the db handle, later dereferences that NULL
  handle* → **null-deref** (oracle `null-deref`). Mapped HIERARCHICAL-REVOKE
  (null-deref flavor): a *failed* close does **not** revoke the live connection,
  yet the wrapper nulls its handle pointer, so a later use faults.
- **Cited artifact**, [expo/expo#34992](https://github.com/expo/expo/pull/34992),
  is a **statement-leak fix** — it swaps a manual `cachedStatements` map for
  `sqlite3_next_stmt()` to finalize all open statements before closing. That is
  resource-management (leak prevention), **explicitly not** a null-deref safety
  fix. So row 12's scenario is currently a **constructed MODEL**, not a lowering
  of #34992.

Full finding: `capstone/benchmarks/sqlite/cve-repros/row12_expo_unfinalized_close/PROVENANCE.md`
(tier already flagged **MODEL / CITATION-MISMATCH**).

This is squarely your lane — you produced the literal matched pairs for rows
3/7/9/5/2/11/14. Row 12 is HIERARCHICAL null-deref flavor; the mechanism you
already have (`revoke_on_free_hier_alloc.h` sub-arena cascade + a *failed*-close
path that does **not** revoke) should express it with no new primitive.

## Deliverable — one of two honest outcomes

**Outcome A (preferred, if the artifact exists): graduate row 12 to ✓ literal.**
Find the upstream artifact that *actually* documents the "ignore `SQLITE_BUSY` on
close → NULL/stale handle → later deref" bug (in **any** binding — as with rows
5/9 we re-anchored HIERARCHICAL to PHP/Ruby, row 12 may re-anchor to whichever
binding really has this bug), then build a **faithful matched pair**:
- host "before": ASan-observable **null-deref** (oracle `null-deref`), reproducing
  the real bug's shape;
- Capstone "after": RTL boot where the post-close deref of the (revoked-parent /
  never-revoked-because-close-failed) handle **faults**, consistent with the
  HIERARCHICAL null-deref mapping.

**Outcome B (if no faithful upstream artifact exists): make the MODEL honest and
label it so the paper can relabel cleanly.** Keep row 12 explicitly **△ modeled**,
update its `PROVENANCE.md` with the negative search result (what you looked for,
where, why nothing matches), and confirm the ledger's caption wording. A clean,
documented MODEL is a fine outcome — the liability is the *false "verified" claim*,
not the existence of a modeled row.

Either way the "All links verified" exposure is removed.

## Fidelity bar (same standard as task-010/012)

- The observed host fault must be a **null-deref** (matching row 12's essence /
  oracle), *not* a UAF — do not silently convert it to the UAF shape.
- For Outcome A, the faulting deref must be on **SQLite's own handle path** (the
  connection/statement the binding nulled or the parent revoked), verified by PC ∈
  the relevant function and `badaddr` in the right region — same evidence standard
  you used for rows 5/9 (`pc`, `vaddr ∈ [fn]`, `badaddr ∈ sub-arena`).
- The interesting fidelity point is the **return-code path**: a *failed*
  `sqlite3_close`/`sqlite3_close_v2` (returns `SQLITE_BUSY`) does **not** revoke
  the connection — model that the wrapper nulls its handle anyway, so the fault is
  a genuine null/stale-handle deref, not a successful-teardown cascade.

## Strict scope (lane rules)

- **Additive only.** Expected: no `llvm/` change and **no `capstone/capstone-qemu`
  gitlink bump** — row 12 is a null-deref via the *existing* hier cascade + a
  no-revoke-on-failed-close path; it should need no emulator change. If you find it
  genuinely does, **stop and coordinate in COORDINATION.md first** (that turns this
  into an emulator-lane task).
- **Do not touch A's files:** `sqlite_row3*`/`sqlite_row2*`/`sqlite_row7*`,
  `revoke_on_free_hier_alloc.h`, `revoke_on_free_alloc.h`, `start.S`, the monitor,
  `capstone-c`, and A's `cve-repros/row*/before*.c` + `oracle*` +
  `run-host-asan-repros.sh` for rows you are not touching. Row 12's own
  `before*.c`/`oracle`/README/PROVENANCE are yours to edit.
- Commit on `capstone-bootstrap-b`, **exact paths**, **no `Co-Authored-By:`**, no
  debug/report files. If you boot QEMU, claim the rootfs lock in COORDINATION.md
  and release it when done.

## Steps

1. **Upstream search (the crux).** Look for a real "ignore `SQLITE_BUSY` on close,
   then deref the nulled/stale handle" bug across bindings — expo-sqlite issues
   (not just PR #34992), better-sqlite3, node-sqlite3, PHP sqlite3, Python
   `sqlite3`, sqlite3-ruby, go-sqlite3. Capture URLs + the exact failing code path.
   Decide Outcome A vs B on the evidence — do not force a match.
2. **Outcome A:** build `sqlite_row12_domain.c` + `run-sqlite-row12.sh` (mirror the
   row-9/row-5 structure), reusing the hier cascade with a *failed-close→no-revoke*
   path; add the host "before" repro; validate `OPT_LEVELS="-O0 -O2"`, fault +
   no-null control each; record PC/vaddr/badaddr evidence.
   **Outcome B:** write the negative-search result into row 12's `PROVENANCE.md`.
3. **Update the records:** row 12 `PROVENANCE.md` (tier → ✓ literal or documented
   △ modeled), `PROVENANCE-LEDGER.md` (row-12 line + tally + the "Open item"),
   `stage2-mapping.md` if the literal set changes, and the row-12 README "after"
   section. (A owns the paper caption; your job is to make the ledger true.)
4. **Regressions:** Capstone lit unchanged/green; the shared allocators untouched
   so rows 3/7 + `hier-revoke-probe` unaffected; any new boots green.
5. **Report** with the outcome (A or B), the upstream evidence, and — as always —
   an explicit line on whether `llvm/` or the `capstone-qemu` gitlink changed
   (expected: neither). Write the trail to
   `history/DD-MM-YYYY_HH-MM-SS_row12-provenance.md`.

## Closing note

This is the last known provenance liability in the corpus table; closing it (as
literal or as an honestly-labeled model) lets the paper's `tab:scope` caption be
reworded from "All links verified" to the accurate per-tier count. Nothing here
is expected to need A or the PI — pure repro + provenance, same as task-010/012.
