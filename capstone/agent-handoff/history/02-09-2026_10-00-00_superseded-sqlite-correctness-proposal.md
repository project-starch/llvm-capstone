# SUPERSEDED: SQLite capability-correctness testing proposal (July, unmerged)

Rescued from the orphan branch capstone-bootstrap-sqlite-qemu before its deletion.
Overtaken by events: SQLite runs, and capstone-sqlite-unpatch carries the live testing work.

# SQL correctness testing under capability enforcement — proposal

**Status: PROPOSED — for review before building. Created 2026-07-29.**

Per the "propose before big directions" norm, this is the doc to review first.

**Goal:** verify SQLite's SQL semantics *at scale, with capability enforcement on*,
inside a Capstone domain. Everything below is about enforcement-on execution; the
hard part is not the SQL, it is that a domain has no operating system.

Companion reading: `ref/HOW-TO-RUN-ON-QEMU.md` (the mechanism),
`design/split-host-enclave-strategy.md` (the architecture),
`design/sqlite-minimal-vfs-path.md` (the file-I/O route),
`design/sqlite-corpus-fidelity-ceiling.md` (the model for stating a ceiling).

---

## 1. Where we start

`bash capstone/benchmarks/sqlite/run-sqlite-memory.sh` is green on QEMU:

```
row name=alpha value=11 / beta 22 / gamma 33
__CAPSTONE_SQLITE_EXTENDED_PASSED__
__CAPSTONE_SQLITE_MEMORY_PASSED__
```

* `sqlite_memory_capstone.dom` — `EM_CAPSTONE (0x103)`, 4.0 MB, built by the
  in-tree Capstone clang, loaded by the monitor (`Created domain ID = 0`), run
  under confinement. Zero faults, zero asserts.
* Workload: `sqlite_capstone_domain.c`, **312 lines** — transactions, a secondary
  `INDEX`, `INTEGER PRIMARY KEY`+`REAL`, bound prepared inserts, `UPDATE`/`DELETE`,
  aggregates + sorter, index-driven `WHERE`, `JOIN`, `GROUP BY`, string functions.
  In-memory only (`SQLITE_OS_OTHER=1`, no VFS).

**This is the entire capability-enforced SQL evidence that exists.** It proves the
engine runs confined. It does not establish that SQL *results* are correct across
any meaningful surface — 312 lines of hand-written assertions is a smoke test.

---

## 2. Why this is hard — the challenges

### C1. A domain has no operating system, so the harness cannot come with it

SQLite's suite is TCL. `testfixture` embeds a TCL interpreter and ~90 harness C
files, and the `.test` scripts assume a filesystem, process control, and an
interactive interpreter. A domain has:

* a **28-function libc shim** (`adapted/capstone_sqlite_libc.h`) — `malloc`/`free`/
  `realloc`, `mem*`, `str*`, `ctype`, `getenv`, `strftime`, `abort`. No `printf`,
  no `fopen`, no syscalls.
* no VFS in the current build (`SQLITE_OS_OTHER=1`).

**Consequence:** the corpus cannot be *interpreted* inside the domain. It must be
**compiled in as data** — a C table of `{sql, expected}` — and executed by a loop
that calls SQLite directly and compares. This is the same shape
`sqlite_capstone_domain.c` already uses, scaled up.

### C2. The engine under test is patched, and the patches are the risk

`build-sqlite-capstone.sh` applies **23 `sed` edits** to `sqlite3.c` before
compiling for `capstone64`. Several are semantic:

| Category | Example | Why it exists |
|---|---|---|
| Ternary selecting between pointers → helper fn | `isView?"view":"table"` → `sqlite3CapstoneViewOrTable(isView)` | backend gap on `select` of two capabilities (6 sites) |
| Static tables holding pointers → built at runtime | `static FuncDef aBuiltinFunc[]` → local + copy loop + `strcmp` chain re-assigning `pUserData` | a capability in static data has no valid tag at load time |
| Static struct of fn pointers → runtime init | `memsys5Methods` | same |
| Alignment | `nByte = SZ_VDBECURSOR(nField)` → `(…+15)&~15` | capabilities are 16 B; `ldc`/`stc` fault when misaligned |
| Pointer subtraction | `(int)(pReadr1 - pMerger->aReadr)` → cursor difference | subtracting two capabilities is not a byte offset |

The `strcmp` chain hand-maps `ltrim`/`rtrim`/`trim`/`max`/`iif`/`likely`/
`unistr_quote` to specific `pUserData` integers. **A wrong mapping silently
changes what those SQL functions do**, and the current 312-line test would never
notice.

**This is the strongest argument for the corpus**: under enforcement-on, the
in-domain corpus is the *only* way to validate these patches, because the patched
engine only runs in a domain. The corpus must therefore be wide enough to cover
the SQL surface the patches touch — builtin functions, the sorter, `ALTER`,
`printf`-family formatting.

**Note for the paper:** SQLite does **not** run unmodified on Capstone. It needs
23 source edits. State that plainly rather than let a reviewer find it.

### C3. Domain image size bounds the batch

The corpus will not fit in one domain image (the domain is already 4.0 MB with a
312-line workload). Cases must be **batched across runs**, with a driver that
sequences batches and aggregates results — mechanically similar to sharding, but
the batching is a property of the domain, not an optimisation.

### C4. Result comparison must be faithful without TCL

Expected values in `.test` files are TCL list renderings (`{11 22}`, quoting and
whitespace rules included). The in-domain comparator is C. Getting float
formatting, NULL rendering, and list quoting to match exactly is fiddly, and
getting it *wrong* produces false failures that look like engine bugs.

### C5. Not all "SQL" cases are self-contained

Some `do_execsql_test` bodies interpolate TCL variables or depend on state set by
surrounding TCL. Those must be **filtered out**, and the filter must be
conservative — silently including a case whose SQL references an undefined
variable yields a spurious failure.

### C6. In-memory only, until the VFS shim lands

Journalling, `ATTACH` to a file, `VACUUM` into a file, and anything touching
durability are untestable under enforcement today. `design/sqlite-minimal-vfs-path.md`
scopes the route — SQLite core → Capstone VFS shim → HostCall file service — and
the file ops are already validated (`FILE_OPEN`, `FILE_READ`, `FILE_WRITE`,
`FILE_SYNC`, `FILE_STAT_BASIC`, `FILE_TRUNCATE`, `FILE_CLOSE`). Not built.

### C7. The C-API half needs a binding that does not exist

The tests that exercise pointer lifetimes across the host↔SQLite boundary — the
capability-interesting ones — call `sqlite3_*` in-process. Putting those behind
the domain boundary is the "Pointer-Safe Marshalling" proposal
(`design/sqlite-marshalling-feasibility.md`): primitives wired C→QEMU, binding not
built. Research project, not on this plan.

---

## 3. The addressable set

Measured over the suite's `.test` files:

| form | count | portable into a domain? |
|---|---|---|
| `do_execsql_test` (SQL in, result out) | 13,976 | **yes** |
| `do_catchsql_test` (SQL in, error out) | 1,417 | **yes**, needs error-code/message comparison |
| `do_test` (arbitrary TCL / C-API / VFS) | 31,200 | **no** — see C7 |

**~15,400 cases are mechanically portable**, minus whatever C5 filtering removes.

---

## 4. Proposed stages, each gated

**Stage 0 — make the corpus source reproducible.**
The suite ships only in the full source tree (`sqlite-src-<ver>.zip`), not the
amalgamation. Add `fetch-sqlite-src.sh` beside the existing `fetch-sqlite.sh`,
pinned to the same version. Small, and every later stage depends on it.

**Stage 1 — extractor + in-domain corpus runner.**
Parse `do_execsql_test` / `do_catchsql_test` into a C table; build a domain that
loops, executes, compares, and reports counts through the shared region; batch per
C3. Follow `ref/HOW-TO-RUN-ON-QEMU.md` §2 and copy `run-sqlite-memory.sh`'s shape.
**Gate:** a first batch runs green end-to-end, with a deliberately-broken expected
value proving the comparator actually fails when it should.

**Stage 2 — scale and report coverage.**
Run the full portable set in batches. Report cases run / passed / filtered-out, and
which SQL features the patched sites (C2) are covered by. **Gate:** any failure is
triaged as engine bug vs comparator bug (C4) vs bad extraction (C5) before the
number is quoted.

**Stage 3 (optional) — file-backed via the minimal VFS shim.**
Extends Stage 2 past in-memory. Deliberately last: Stage 2 delivers the headline
claim without it.

---

## 5. The ceiling — state it, don't paper over it

Following `design/sqlite-corpus-fidelity-ceiling.md`, which scores the CVE corpus
as 5 LITERAL / 12 probes and states a realistic ceiling of **~14/17 rather than
17/17**, on the reasoning that *a capability lifetime can only enforce a real
pointer-lifetime event*.

The same discipline applies here:

> **The SQL-semantics half of SQLite's suite is portable under enforcement; the
> C-API/VFS half is not.** ~15,400 of ~46,600 cases.

The remainder is not blocked by effort — those tests target the *implementation*
(6,161 `sqlite3_*` C-API test commands, 1,163 file/VFS manipulations), and reaching
them needs C7's binding or full hosted userspace, which
`design/split-host-enclave-strategy.md` explicitly recommends against.

Saying this is a stronger claim than an unqualified "we ran the suite."

---

## 6. Rejected alternative — run the stock suite on the platform

`testfixture` can be cross-built for riscv64 and run in guest Linux on
`virt-capstone`. This was tried and works, but it is **not** what we want: that
binary is `EM_RISCV`, built with the buildroot cross-gcc, running as an ordinary
Linux process. SQLite there holds no capabilities and sits in no domain, so
nothing about its memory accesses is checked. It exercises the CPU model, not
enforcement — and its numbers must never be presented as a capability result.

Recorded here only so the question does not get re-asked.

---

## 7. Open questions for reviewers

1. **Is the whole ~15,400 the right target**, or is a curated few-hundred subset —
   chosen to cover the SQL surface the 23 patches touch (C2) — better value?
2. **Is C2 the real motivation?** This proposal argues the corpus's primary job is
   validating the patched engine under enforcement. If reviewers disagree, the
   coverage target changes.
3. **Does the ceiling in §5 need the project lead's sign-off** before it reaches
   the paper? It is a limiting statement about what enforcement can be tested
   against.
4. **Stage 3 priority** — is file-backed SQL under enforcement needed for the
   paper's claims, or is in-memory sufficient?
5. **Is C4 worth the effort**, or should the corpus compare against results
   captured from a reference run rather than the `.test` files' TCL renderings?
