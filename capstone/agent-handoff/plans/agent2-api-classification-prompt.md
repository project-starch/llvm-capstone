# Agent 2 task: SQLite C API boundary classification

> **This whole file is your prompt.** Self-contained: you need only to read a C
> header (`sqlite3.h`), reason about each function's data-flow direction, and
> write a Markdown/CSV table. No capabilities, RISC-V, QEMU, or compiler knowledge.
> You did the SQLite defect-reproduction corpus previously; this is the companion
> task.

---

## Your role

Classify the **entire public SQLite C API** by how data crosses the
host↔engine boundary. The output is a reference table for a research paper. This
is a reading-and-tabulating task, not a coding task.

## Background you need (short)

SQLite is a C library embedded into a host application; the host calls it through
functions declared in `sqlite3.h`. Every function that moves a **pointer, buffer,
handle, or callback** across that boundary carries a *lifetime or ownership
obligation*. We are cataloguing the whole surface so we can show which capability
mechanism enforces each obligation.

## The four boundary directions (classification buckets)

Assign **exactly one** primary bucket per function:

- **`H->E`** (Host → Engine): the host passes data/buffers/SQL/parameters *into*
  the engine. Examples: `sqlite3_bind_blob/text/int`, `sqlite3_exec` (SQL text in),
  `sqlite3_prepare_v2` (SQL text in), `sqlite3_result_*` (a UDF returning a value
  into the engine), `sqlite3_deserialize` (hands the engine a memory image).
- **`E->H`** (Engine → Host): the engine hands borrowed data *out* to the host.
  Examples: `sqlite3_column_blob/text/value`, `sqlite3_value_blob/text` (UDF arg
  read), `sqlite3_column_name`, `sqlite3_errmsg`, `sqlite3_serialize`.
- **`E<->H`** (Bidirectional / handle threaded both ways): a handle or pointer
  that crosses in both directions or is co-owned. Examples:
  `sqlite3_bind/result/value_pointer` (a raw typed pointer threaded through SQL),
  `sqlite3_blob_open/read/write` (incremental BLOB handle), lifecycle handles
  (`sqlite3_prepare`/`step`/`reset`/`finalize`, `sqlite3_open`/`close_v2`,
  `sqlite3_backup_*`).
- **`CB`** (Callback / hook): host code the engine *stores and invokes later*,
  with a context + optional destructor lifetime. Examples:
  `sqlite3_create_function[_v2]`, `sqlite3_set_authorizer`,
  `sqlite3_progress_handler`, `sqlite3_busy_handler`, `sqlite3_trace_v2`,
  `sqlite3_commit/rollback/update_hook`, `sqlite3_create_collation[_v2]`,
  `sqlite3_wal_hook`, `sqlite3_preupdate_hook`.
- **`N/A`** (no boundary pointer): administrative / value-only functions that move
  no borrowed pointer, buffer, handle, or callback. Examples: `sqlite3_libversion`,
  `sqlite3_config` (mostly), `sqlite3_changes`, `sqlite3_errcode`,
  `sqlite3_threadsafe`, `sqlite3_sleep`. Bucket these `N/A` and move on — do not
  agonize.

If a function plausibly fits two buckets, pick the one describing its **most
lifetime-sensitive** transfer and note the secondary in the Notes column.

## The obligation + primitive columns

For each **non-`N/A`** function add:

- **Obligation**: a short phrase — the lifetime/ownership rule the host must
  uphold. E.g. "borrowed ptr, invalid after next step/reset/finalize";
  "ownership flag STATIC/TRANSIENT/destructor"; "callback ctx must outlive
  registration"; "handle invalidated by parent close".
- **Primitive**: which capability mechanism would enforce it — one or more of:
  - **L** linear (exclusive, move-only borrow),
  - **R** revocation (explicit revoke at the borrow's end),
  - **H** hierarchical revoke (a senior revoke cascades on close/destroy),
  - **U** uninitialised (no read authority until initialised),
  - **S** sealed (safe callback domain switch).
  Rules of thumb: borrowed out-ptrs (`E->H`) → **R** (row-scoped) or **L,R**;
  bound-in buffers / callbacks (`H->E`, `CB`) → **L,R** (+**S** for `CB`);
  lifecycle/parent-child handles (`E<->H`) → **H**; reclaimed buffers / use-before-
  init → **U**. (These mirror the paper's Table 4; approximate, the supervising
  agent refines.)

## How to enumerate the API

```bash
source capstone/tests/capstone-test-env.sh
bash capstone/benchmarks/sqlite/fetch-sqlite.sh    # if not already fetched
SQLITE_DIR="$CAPSTONE_TMP_ROOT/sqlite-src/sqlite-amalgamation-3530300"
grep -nE 'SQLITE_API .* sqlite3_[a-z0-9_]+\(' "$SQLITE_DIR/sqlite3.h" | head
```

Extract every **public** `sqlite3_*` function prototype from `sqlite3.h`. Group the
`_v2`/`_v3`/`16` (UTF-16) and type-suffixed variants (`_blob`/`_text`/`_int`/
`_int64`/`_double`/`_null`) onto **one row** with the base name where their boundary
behavior is identical (e.g. one `sqlite3_bind_*` row, not eight) — note the variants
in the Notes column. This keeps the table readable; aim for ~60–90 rows, not 300.
Skip experimental (`SQLITE_EXPERIMENTAL`) and deprecated (`SQLITE_DEPRECATED`)
functions, but note that you skipped them.

## STRICT ground rules

- **Only create/modify** `capstone/benchmarks/sqlite/cve-repros/api-classification.md`
  and `capstone/benchmarks/sqlite/cve-repros/api-classification.csv`. Touch nothing
  else — not `sqlite3.h`, not the corpus, not the compiler, not build scripts.
- **Do NOT commit.** Leave the files; the supervising agent reviews and commits.
- **Do not invent functions.** Every row must correspond to a real prototype in
  `sqlite3.h`. If unsure of a function's direction, put your best guess and add
  "(uncertain)" in Notes rather than dropping it.
- If the header has hundreds of functions, prioritise the data/lifetime-relevant
  ones (bind/column/value/result/blob/pointer/prepare/step/finalize/close/backup/
  serialize + all hooks/callbacks). Bulk-bucket the clearly administrative rest as
  `N/A`.

## Output format

`api-classification.md` — a Markdown table, sorted by Direction then name:

```
# SQLite C API boundary classification

Source: sqlite3.h (SQLite 3.53.3). Directions: H->E host→engine, E->H engine→host,
E<->H bidirectional/handle, CB callback, N/A no boundary pointer.

| Function (base) | Group | Direction | Obligation | Primitive | Notes/variants |
|-----------------|-------|-----------|------------|-----------|----------------|
| sqlite3_bind_*  | bind  | H->E | ownership flag STATIC/TRANSIENT/destructor | L,R | _blob/_text/_int/_int64/_double/_null/_zeroblob |
| sqlite3_column_* | column | E->H | borrowed; invalid after step/reset/finalize/type-convert | R | _blob/_text/_value/_name; _text16 variants |
| sqlite3_create_function[_v2] | udf/cb | CB | ctx+destructor must outlive registration | L,R,S | v2 adds xDestroy |
| ... | | | | | |

## Summary
Counts per direction: H->E = N, E->H = N, E<->H = N, CB = N, N/A = N. Total rows = N.
```

Also emit the same rows as `api-classification.csv` (header:
`function,group,direction,obligation,primitive,notes`) for downstream tooling.

## Cross-check (do this at the end)

The 18 reproduced defects in `cve-repros/row*/README.md` each abuse a specific API.
Confirm every API they name appears as a row in your table (e.g. `column_name`,
`progress_handler`, `create_function`, `set_authorizer`, `finalize`, `close`,
`backup_init`, `bind_*`, `column_type`). List any that are missing in a
"Coverage" note at the bottom.

## When done

Return: the summary counts, the total row count, and any uncertain/missing entries.
Do not commit. Do not do any Capstone/RISC-V work.
