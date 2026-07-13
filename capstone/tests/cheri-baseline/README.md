# cheri-baseline — CHERI security baseline for the corpus (agentB-015)

The paper's #1 security baseline is **CHERI**. For each corpus CVE the storyline
is: *not blocked under the baseline, blocked on our system.* This directory
produces the **empirical CHERI column** — compile each row's minimal vulnerable
shim as CHERI-RISC-V **purecap** and run it under CHERI-QEMU, classifying whether
CHERI catches the defect and *when*. It is **measurement + classification**, not
a system build, and it is the **CHERI baseline, not our system**.

**Results and the config reality live in `RESULTS.md`.**

## The thesis under test

CHERI's temporal safety is **`free()`-triggered and asynchronous** (the
Cornucopia / CHERIvoke quarantine-and-sweep model): freed allocations are
quarantined and a later **stop-the-world sweep** invalidates dangling
capabilities. Our corpus defects occur at lifecycle contract points
(`step`/`reset`/`finalize`/`close`); several leave the memory **still allocated**
(logically invalid, not freed). Prediction: CHERI either **misses** them
(stale-but-allocated: the capability is in-bounds and un-revoked) or catches them
**only after a sweep, not synchronously at the boundary**. A refutation is a
valid, important result and is reported as such.

## Three configs (one boot)

| Config | Revocation | Meaning |
|---|---|---|
| `spatial`  | OFF | CHERI spatial safety only (bounds + tags). |
| `temporal` | ON, async quarantine | **realistic** CHERI temporal deployment. |
| `eager`    | ON, revoke on every `free` | aggressive **synchronous** upper bound. |

Toggled at runtime via `security.cheri.runtime_revocation_default` and
`security.cheri.runtime_revocation_every_free_default`; each process's actual
policy is confirmed with `malloc_revoke_enabled()` (see `cheri_status.c`).

## Verdict taxonomy

- **BLOCKED-SYNC** — faults in `spatial`: caught *at* the offending access with no
  revocation (null-deref / out-of-bounds).
- **BLOCKED-SWEEP** — no spatial fault, but a revocation config faults: the
  dangling capability is caught only by a sweep, **not at the contract point**.
  `(async)` = caught even in the realistic default; `(eager)` = only with
  revoke-on-every-free.
- **MISS** — survives every config: the defect is not caught.

## Real SQLite does not run purecap here — the shims link a lifecycle harness

Upstream SQLite faults purecap in a standalone binary *before* reaching any
injected defect (`SIGBUS`/`BUS_ADRALN` in `sqlite3_open` at `THREADSAFE=0`;
deadlock at `THREADSAFE=1`), for both the vanilla amalgamation and CheriBSD's
patched `contrib/sqlite3` — see `RESULTS.md`. Since the CHERI verdict depends only
on each defect's memory-lifecycle events (which handle is freed/reused/null when,
and whether the offending access dereferences that memory), each corpus
`before.c` is compiled **VERBATIM** against `mock-sqlite/`, a minimal harness that
reproduces exactly those events. The shims are unmodified; the harness runs clean
purecap (`sanity_mock`). Binaries are built `-O0` so the UB dangling access is
actually emitted (`-O1+` hoists/elides it).

## Files

| File | Role |
|---|---|
| `rows.tsv` | maps the paper's 15-row table onto the on-disk `cve-repros` dirs (which still use the pre-trim 19-row numbering) + oracle + prediction. |
| `mock-sqlite/` | minimal SQLite-lifecycle harness (`sqlite3.h` + `mock_sqlite3.c`) the shims link against; reproduces the alloc/free/callback/invalidation events, not SQL. |
| `compile-purecap.sh` | builds the mock + one purecap ELF per row (shims verbatim, `-O0`), plus sanity probes (auto-probes the CHERI `-march`). |
| `sanity_clean.c` | defect-free SQLite exercise; built 3 ways — `sanity_vanilla`/`sanity_clean` (real upstream/patched amalgamation, fault) and `sanity_mock` (the harness, clean). |
| `row3_reuse.c` | faithful *reuse-not-free* variant of the diesel defect (the headline stale-but-allocated case). |
| `diag.c` | signal-catching probe that pinpointed the real-SQLite fault (step marker + `si_code`/`si_addr`). |
| `run-in-guest.sh` | guest-side: sets the config's sysctls, runs each ELF (under `timeout`), prints structured result lines. |
| `cheri_status.c` | prints `malloc_revoke_enabled()` so config reality is recorded, not assumed. |
| `cheri-run.py` / `oneshot.py` | host-side pexpect drivers: boot CheriBSD purecap, run the 3 configs (or one command), capture the serial log. |
| `classify.py` | parses the serial log into the per-row table + verdict. |
| `run-cheri-baseline.sh` | end-to-end: compile → bake overlay into the disk image → boot → classify. |

## Vehicle

CHERI-RISC-V **purecap** on CHERI-QEMU, built with
[`cheribuild`](https://github.com/CTSRD-CHERI/cheribuild) **outside** the llvm
tree (default source root `~/cheri`; this workspace uses `~/cheri-ws/cheribuild`).
QEMU is the right vehicle here: we test **catch / no-catch**, not performance.
Toolchain: CHERI-LLVM (clang 17, CTSRD-CHERI), CheriBSD purecap,
`qemu-system-riscv64cheri` 7.1.0.

## Run

```bash
capstone/tests/cheri-baseline/run-cheri-baseline.sh
```

No *our-QEMU* rootfs lock is needed — CHERI has its own QEMU and image.
