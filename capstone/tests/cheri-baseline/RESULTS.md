# CHERI baseline — corpus catch/no-catch results (agentB-015)

**This is the empirical CHERI column of the paper's security table (Lane A).**
For each of the 15 in-scope corpus defects it records whether CHERI-RISC-V
**purecap** catches the defect's dangling-pointer access, and *when*, under three
revocation configurations. It is measurement + classification, the **CHERI
baseline, not our system**.

## Toolchain / vehicle (config reality — state this in the paper)

| Component | Version |
|---|---|
| CHERI-LLVM (clang) | 17.0.0, CTSRD-CHERI `7e122876ee01` |
| CheriBSD (kernel + purecap userspace) | FreeBSD 15.0-CURRENT `CHERI-PURECAP-QEMU`, INVARIANTS+WITNESS+**CHERI_CAPREVOKE** |
| Emulator | `qemu-system-riscv64cheri` 7.1.0 (CTSRD-CHERI) |
| Build | `cheribuild` (out-of-tree, `~/cheri-ws/cheribuild`); shims compiled purecap `-O0` |

**Heap temporal safety is available AND on by default here:**
`security.cheri.runtime_revocation_default = 1`. We toggle it (and
`runtime_revocation_every_free_default`) to isolate three configs, and confirm
each process's actual policy with `malloc_revoke_enabled()`.

| Config | `runtime_revocation_default` | `..._every_free_default` | `malloc_revoke_enabled()` | meaning |
|---|:--:|:--:|:--:|---|
| **spatial**  | 0 | 0 | 0 | CHERI spatial safety only (bounds + tags) |
| **temporal** | 1 | 0 | 1 | revocation ON, async quarantine — **realistic** CHERI temporal deployment |
| **eager**    | 1 | 1 | 1 | revocation ON, revoke on **every** `free` — aggressive synchronous upper bound |

## Why the shims link against a lifecycle harness, not real SQLite (read this)

**Upstream SQLite does not execute under CHERI purecap in a standalone binary
here**, so it cannot be the vehicle for measuring the *injected* defect:

- Compiled `THREADSAFE=0`, the amalgamation faults **before reaching any injected
  bug** — `SIGBUS` / `BUS_ADRALN` (misaligned-capability access) inside
  `sqlite3_open` (fault addr `0x149f86`, ≡ 6 mod 16). Both the vanilla upstream
  amalgamation and CheriBSD's *patched* `contrib/sqlite3` (3.46.1) fault
  identically — the `sanity_vanilla` and `sanity_clean` probes both die with 138.
- Compiled `THREADSAFE=1` (CheriBSD's own build, incl. their prebuilt purecap
  object, dynamic **and** static), the process **hangs** in a standalone binary.

CheriBSD's base uses `libsqlite3` (via `kyua`) purecap, so it *is* achievable
inside their full build system, but porting upstream SQLite to run purecap
standalone is a project in itself and orthogonal to this task's question.

The CHERI verdict for every corpus row depends **only** on the defect's
memory-lifecycle events — which handle is freed / reused / null when, and whether
the offending access dereferences that memory — not on SQLite's SQL engine. So we
compile **each corpus `before.c` VERBATIM** against a minimal SQLite-lifecycle
harness (`mock-sqlite/`) that reproduces exactly those events: `open`/`prepare`
allocate handles, `close`/`close_v2` free the connection, `finalize` frees the
statement (a 2nd `finalize` is a real double-free), `exec` invokes the registered
progress/UDF/authorizer callback, `step`/`reset` dereference the (possibly freed)
connection, `column_name` returns a statement-owned buffer that `finalize` frees.
The harness itself runs clean purecap (`sanity_mock` exits 0); the shim sources
are unmodified.

**Optimisation:** the shims are built `-O0` on purpose. A use-after-free / null
deref is undefined behaviour; at `-O1+` the compiler hoists the handle load
before the `free` or elides the dangling access, so the access we want CHERI to
police is never emitted. `-O0` emits every load/store as written — the faithful
condition for a catch/no-catch measurement.

## Results (15 rows + one supplementary; deterministic)

Verdict taxonomy: **BLOCKED-SYNC** = faults in `spatial` (caught at the access
with no revocation); **BLOCKED-SWEEP** = no spatial fault but a revocation config
faults (caught only by a sweep, *not at the contract point*; `(async)` = even the
realistic default, `(eager)` = only revoke-every-free); **MISS** = survives all.
Exit codes: `SIGPROT`(162) = capability fault, `SIGABRT`(134) = allocator
double-free abort, `exit0` = ran to completion (defect survived).

Sanity (must hold or the row data is invalid): `sanity_vanilla` **138**
(SIGBUS, upstream SQLite faults purecap), `sanity_clean` **138** (patched SQLite
faults too), `sanity_mock` **0** (the lifecycle harness runs clean).

| Row | Defect (class) | spatial | temporal (async, default) | eager (revoke/free) | CHERI verdict |
|----|----------------|:------:|:------:|:------:|----|
| 1 | CPython gh-142830 progress ctx freed mid-call (UAF) | MISS | MISS | SIGPROT | blocked only *post-free*, not at the callback |
| 2 | rusqlite hook closure (UAF) | MISS | MISS | SIGPROT | blocked only post-free |
| 3 | diesel column ptr cached across `step` (UAF as shimmed) | MISS | MISS | SIGPROT | blocked only post-free — but see `3r` |
| 4 | PHP stmt after `close` (use-after-close) | MISS | MISS | SIGPROT | blocked only post-free |
| 5 | PHP stmt/db destruction order (UAF) | MISS | MISS | SIGPROT | blocked only post-free |
| 6 | PHP UAF via UDF | MISS | MISS | SIGPROT | blocked only post-free |
| 7 | CPython gh-99886 cursor dealloc (UAF) | MISS | MISS | SIGPROT | blocked only post-free |
| 8 | CPython backup on closed conn (use-after-close) | MISS | MISS | SIGPROT | blocked only post-free |
| 9 | sqlite3-ruby finalize after db-free (UAF) | MISS | MISS | SIGPROT | blocked only post-free |
| 10 | sqlite3-ruby stmt reuse after close (use-after-close) | MISS | MISS | SIGPROT | blocked only post-free |
| 11 | go-sqlite3 double-free | MISS | **SIGABRT** | SIGPROT | blocked by the **allocator's double-free abort** (not a sweep) |
| 12 | expo unfinalized-stmt NULL on close | **SIGPROT** | SIGPROT | SIGPROT | **BLOCKED-SYNC** (null-cap tag fault) |
| 13 | CPython null-deref deleted row_factory | **SIGPROT** | SIGPROT | SIGPROT | **BLOCKED-SYNC** (null-cap tag fault) |
| 14 | CPython uninitialised Connection | **SIGPROT** | SIGPROT | SIGPROT | **BLOCKED-SYNC** (null-cap tag fault) |
| 15 | datasette authorizer ctx lifetime (UAF) | MISS | MISS | SIGPROT | blocked only post-free |
| **3r** | **diesel *reuse-not-free* (real defect)** | **MISS** | **MISS** | **MISS** | **never blocked** — stale-but-allocated, cap stays tagged & in-bounds |

Exit codes: SIGPROT=162 (capability fault), SIGABRT=134 (allocator double-free
abort), MISS=exit 0 (or the returned stale byte, e.g. `3r`=114=`'r'`).

**Tally:** spatial-only blocks **3/15** (the null-derefs). Async-default temporal
blocks **4/15** (null-derefs + double-free-abort) and **0/10 use-after-free** at
the contract point. Revoke-every-free blocks **15/15** of the shim rows — but the
real reuse-not-free defect (`3r`) is blocked by **no** configuration.

### vs the task's predicted oracle

Confirmed, with two refinements to surface:
- The oracle's config B ("+revocation sweep") splits: the **realistic async
  default catches none of the use-after-free rows** (stronger than "post-sweep
  only"); they are caught **only** under the non-default, expensive
  revoke-on-every-free. So base CHERI as normally deployed does **not** block the
  temporal class at the lifecycle contract point.
- Row 3: the corpus `before.c` frees the column buffer via `finalize`, so it is
  caught under eager; but the **real diesel defect reuses the buffer in place
  without freeing** (`3r`), and that is MISS in every config — the clean
  "CHERI-can't" the paper wants. Row 11's double-free is caught by the allocator's
  abort, not by revocation.


## What the data says (for the paper)

1. **Base CHERI purecap (spatial only) is blind to the entire temporal class.**
   All 12 use-after-free / use-after-close / reuse / double-free rows run to
   completion; only the 3 null-dereferences trap (the tag check on a null
   capability). This is the column reviewers will focus on: for the corpus's
   temporal defects, **spatial CHERI does not block them**.

2. **Realistic CHERI temporal safety (async revocation, the default) still does
   not catch these at the contract point.** With revocation ON but sweeping on
   quarantine pressure, the short-lived reproducers free and reuse before any
   sweep runs, so the use-after-free rows **still MISS** — the dangling
   capability is revoked only by a later stop-the-world sweep, never
   synchronously at `step`/`reset`/`finalize`/`close`. The only temporal defect
   caught here is the **double-free** (the allocator aborts on the second free,
   independent of the sweep). This is the paper's thesis, confirmed.

3. **Even revoke-on-every-free (the expensive, non-default upper bound) catches
   these only *after* the free, not at the logical-invalidation point** — and the
   one defect it cannot catch at all is the **reuse-not-free** case (`row3_reuse`,
   the real diesel pattern): the buffer is overwritten in place, never freed, so
   the capability stays tagged and in-bounds and **no CHERI configuration**
   catches the stale read. This is the clean "CHERI can't" headline.

**Bottom line for the security table:** CHERI blocks the corpus's null-deref rows
synchronously and its double-free via the allocator; it blocks the use-after-free
rows only asynchronously (post-sweep, not at the contract point) and only with
revocation enabled; and it **cannot** block the stale-but-allocated reuse case at
all. Our system faults synchronously at the lifecycle contract point in every
row (see the paper's "Our system" column).

## Reproduce

```bash
capstone/tests/cheri-baseline/run-cheri-baseline.sh    # compile -> image -> boot -> classify
```
Needs the cheribuild CHERI stack (see README). No *our-QEMU* rootfs lock.
