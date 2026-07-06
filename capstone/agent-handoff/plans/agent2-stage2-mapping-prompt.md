# Agent 2 task: Stage-2 mapping for the SQLite defect corpus

> **This whole file is your prompt.** It is a reading-and-tabulating task: read a
> few short files already in the repo and produce one Markdown table. No coding,
> no capabilities/RISC-V/QEMU knowledge, no builds, no running anything.

---

## Your role

You previously built the SQLite defect reproduction corpus and the API
classification. This task connects them: for each reproduced defect, decide **how
its Capstone "after" demonstration should be shaped**, so the supervising agent
can generate the runtime probes efficiently. You only produce a table; you do not
build the probes.

## What already exists (read these)

- `capstone/benchmarks/sqlite/cve-repros/row*/README.md` — the 19 defects, each
  with its Class and one-line Essence (you wrote these).
- `capstone/benchmarks/sqlite/cve-repros/api-classification.csv` — direction +
  primitive per API (you wrote this).
- `capstone/tests/runtime-qemu/sqlite-borrow-revoke-probe/README.md` — the ONE
  validated Stage-2 template (row 3): a borrow that is **revoked** at a lifecycle
  point, after which the cached access faults. Read its "Mapping" table.

## The five Stage-2 shapes

Every applicable row maps to exactly one of these (from the paper's Table 4
primitives L/R/H/U/S):

- **BORROW-REVOKE** (primitive R, sometimes L,R) — a borrowed value used past a
  lifecycle point. This is the validated row-3 template. Differs per row only in
  **which lifecycle call ends the borrow**: `step`, `reset`, `finalize`, or an
  implicit type conversion.
- **HIERARCHICAL-REVOKE** (primitive H) — a handle under a parent (statement under
  connection); the **parent close/destroy** cascades the revoke. Use for
  use-after-close / destruction-order rows.
- **LINEAR** (primitive L) — a move-only handle that a second operation cannot
  consume again. Use for **double-free / double-finalize**.
- **UNINIT** (primitive U) — a capability with no read authority until initialised.
  Use for **use-before-initialisation** rows.
- **SEALED-CALLBACK** (primitive S, with L,R) — a callback context borrowed across
  a domain crossing; revoked if freed mid-call. Use for **callback/hook** rows
  where the context is freed while registered or mid-invocation.
- **N/A** — rows with no pointer-lifetime defect (concurrency/liveness, or a purely
  logical stale-state read that has no capability fix). Mark and explain briefly.

## Output — one file

Create **`capstone/benchmarks/sqlite/cve-repros/stage2-mapping.md`**: a Markdown
table, one row per `row*/` directory (19 rows), columns:

| Row id | Class | Stage-2 shape | Lifecycle point that revokes | Primitive | Reuses row-3 template? | Note |
|--------|-------|---------------|------------------------------|-----------|------------------------|------|

- **Stage-2 shape**: one of the six above.
- **Lifecycle point that revokes**: the exact SQLite call at which the borrow
  ends (e.g. `sqlite3_step`, `sqlite3_finalize`, `sqlite3_close_v2`,
  hook-unregister, `sqlite3_reset`). `n/a` for LINEAR/UNINIT/N/A shapes.
- **Primitive**: copy from the defect's API row in `api-classification.csv` where
  applicable; otherwise from the shape.
- **Reuses row-3 template?**: `yes` if it is a BORROW-REVOKE that differs from row
  3 only by the lifecycle call; `no` otherwise (with the reason in Note).

Add a short summary under the table: counts per Stage-2 shape, and an explicit
list of which rows the supervising agent can generate by cloning the row-3 probe
(the `yes` rows).

## Guidance for the tricky rows

- Rows **1, 2, 6, 16** (callbacks/hooks: progress handler, UDF closure, authorizer)
  → SEALED-CALLBACK (context freed while registered / mid-call).
- Rows **4, 8, 10, 12** (use-after-close / backup-on-closed) → HIERARCHICAL-REVOKE
  (parent `close` cascades).
- Rows **5, 7, 9** (destruction-order / cursor dealloc / finalize-after-db-free)
  → HIERARCHICAL-REVOKE (out-of-order teardown of parent/child).
- Row **3, 13, 18, 19** (borrowed value used past step/reset) → BORROW-REVOKE
  (row-3 template; 13/18/19 revoke at `reset`/re-`step` or delete of the borrowed
  handle). Note 18/19 were logical-stale in Stage 1.
- Row **11** (double-finalize) → LINEAR. Row **14** (uninit connection) → UNINIT.
- Row **15** (GIL) → N/A. Row **17** (bind typecast abort) → N/A or a note that a
  bounded read-only borrow only turns the abort into a local fault (weak fit).

These are guidance, not gospel — if a row's README essence suggests a better fit,
use it and say why in the Note.

## STRICT ground rules

- **Only create** `capstone/benchmarks/sqlite/cve-repros/stage2-mapping.md`. Touch
  nothing else.
- **Do NOT commit.** Leave the file; the supervising agent reviews and commits.
- Do not build or run anything. This is a desk task.

## When done

Return the summary (counts per shape + the list of `yes` template-clonable rows).
