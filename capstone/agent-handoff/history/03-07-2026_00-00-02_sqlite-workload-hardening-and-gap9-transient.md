# SQLite workload hardening + gap 9 (SQLITE_TRANSIENT client-API inconsistency)

*Status: 2026-07-03. After SQLite reached a minimal end-to-end pass (gaps 1–8),
the domain test was extended to a richer SQL workload to (a) strengthen the
real-world claim and (b) flush remaining capability gaps. Result: the extended
workload passes; one new gap (9) surfaced — a client-API artifact of the build's
SQLITE_TRANSIENT patch, worked around and documented here.*

## Extended workload (all passing, pure-capability domain)

`sqlite_capstone_domain.c` now runs, after the base 3-row CREATE/INSERT/SELECT:
- a transaction (`BEGIN`/`COMMIT`);
- a second table with an `INTEGER PRIMARY KEY` (rowid btree) + `REAL` column;
- a **secondary `INDEX`** (index-btree cursors — more cursor machinery than the
  base test, which is what surfaced gap 8);
- **bound prepared inserts** (`sqlite3_bind_int/text/double`);
- `UPDATE` and `DELETE`;
- **aggregates + the sorter** (`COUNT`/`SUM`/`MAX`, `ORDER BY ... DESC`);
- an index-driven `WHERE` with comparison operators (exercises the gap-7
  `sqlite3aLTb/aEQb/aGTb` tables);
- a **`JOIN`** across two tables;
- `GROUP BY`;
- **string functions** (`upper`/`length` → `sqlite3UpperToLower`).

Each step is validated; the domain emits `__CAPSTONE_SQLITE_EXTENDED_PASSED__`
before the existing `__CAPSTONE_SQLITE_MEMORY_PASSED__`. `run-sqlite-memory.sh`
gates on both. Confirmed with pristine QEMU (submodule clean). No new alignment or
tag-loss gaps appeared across this machinery — a strong signal the capability port
is broadly correct, not just on the minimal path.

## Gap 9 — SQLITE_TRANSIENT client-API inconsistency

First extended attempt aborted with `cs.cjalr requires capability in rs1`. A QEMU
diagnostic (call-site pc + target; since reverted) pinned it to
**`vdbeMemClearExternAndSetNull`** calling a Mem cell destructor `p->xDel(p->z)`
whose value was **`0xffffffffffffffff` (-1)** — the `SQLITE_TRANSIENT` sentinel,
invoked as a function.

Root cause: the build's `sed` patch replaces `#define SQLITE_TRANSIENT
((sqlite3_destructor_type)-1)` with a real function `sqlite3CapstoneTransient`
**only inside the amalgamation `sqlite3.c`** (to dodge a clang constant-evaluator
crash on `(fnptr)-1` in constant contexts). But the **public `sqlite3.h`** — which
the domain client includes — still defines `SQLITE_TRANSIENT` as `-1`. So a client
`sqlite3_bind_text(..., SQLITE_TRANSIENT)` passes `-1`; the patched core does not
recognize `-1` as its transient sentinel (it expects `sqlite3CapstoneTransient`),
stores `-1` as a real destructor with `MEM_Dyn`, and later **calls `-1`** → `cjalr`
on an untagged pointer.

This is not a core capability-safety gap; it is a faithfulness bug in the
TRANSIENT patch (the `.h` and `.c` sentinels disagree). Any client using the
standard `SQLITE_TRANSIENT` binding hits it.

### Workaround (applied) + proper-fix direction

- **Applied:** the domain test binds persistent (static) buffers with
  `SQLITE_STATIC`, which is the correct destructor for non-owned storage and
  avoids the sentinel entirely.
- **Proper fix (follow-up):** make the patched core treat the public `-1` as
  transient too — patch the single copy-vs-store decision in
  `sqlite3VdbeMemSetStr` (and any peer) to accept both `sqlite3CapstoneTransient`
  and `(sqlite3_destructor_type)-1`; or, better, resolve the underlying clang
  constant-evaluator crash on `(fnptr)-1` so no sentinel substitution is needed at
  all (then `SQLITE_TRANSIENT` stays `-1` end to end). Until then, clients on this
  port must use `SQLITE_STATIC`/a real destructor, not `SQLITE_TRANSIENT`.

## Non-blocking follow-ups still open (unchanged)

- QEMU should deliver in-domain cap faults cleanly (today `riscv_cpu_do_interrupt`
  asserts `env->priv < PRV_C` and aborts) — that abort is also what makes buffered
  domain output invisible on a mid-run fault.
- The 8-byte-alignment class (gaps 6/8) may still surface on paths this workload
  does not reach.
