# Task 015 — CHERI corpus test (paper "Task 1", security baseline)

**Date:** 2026-07-13
**Branch:** `capstone-bootstrap-b`
**Deliverable:** the empirical **CHERI column** of the paper's security table —
for each corpus CVE, does base CHERI block this vulnerability class, and when.
Measurement + classification, not a system build; the **CHERI baseline, not our
system**.

## Vehicle (built out-of-tree via cheribuild)

Full CHERI-RISC-V **purecap** stack, built from scratch in `~/cheri-ws` (nothing
vendored into `llvm/`):

- **CHERI-LLVM** clang 17.0.0 (CTSRD-CHERI `7e122876ee01`).
- **CheriBSD** purecap FreeBSD 15.0-CURRENT `CHERI-PURECAP-QEMU`
  (INVARIANTS+WITNESS+**CHERI_CAPREVOKE**), world+kernel+disk image.
- **`qemu-system-riscv64cheri`** 7.1.0 (CTSRD-CHERI).

Host-build friction resolved without sudo: three missing `-dev` packages
(`libattr1-dev`, `libcap-ng-dev`, `libarchive-dev`) were `apt-get download`ed and
extracted into a local prefix with rewritten `.pc` files (`~/cheri-ws/local-deps`,
helper `add-dev-dep.sh`); QEMU virtfs needed `CPATH`/`LIBRARY_PATH` for the bare
`has_header` check; the CheriBSD **test suite** (`capsicum-test`) fails to build
purecap so world is built `--cheribsd/no-build-tests`; the disk image is built
`--disk-image/no-include-gdb` (gdb needs `makeinfo`, absent); boot needs the
`bbl-baremetal-riscv64-purecap` firmware.

## Config reality (the crux of the comparison)

Heap temporal safety is present and **on by default**:
`security.cheri.runtime_revocation_default = 1`. Three configs, each confirmed
per-process via `malloc_revoke_enabled()`:

- **spatial** — revocation OFF: CHERI spatial safety only.
- **temporal** — revocation ON, async quarantine: the realistic default.
- **eager** — revocation ON + `runtime_revocation_every_free_default=1`: revoke on
  every free, the aggressive synchronous upper bound.

## Why the shims link a lifecycle harness, not real SQLite

**Upstream SQLite does not run under CHERI purecap in a standalone binary here**
— it faults before reaching any injected defect:

- `THREADSAFE=0`: `SIGBUS`/`BUS_ADRALN` (misaligned capability) inside
  `sqlite3_open` (fault addr `0x149f86`), for BOTH the vanilla amalgamation and
  CheriBSD's patched `contrib/sqlite3` 3.46.1. Proven with a step-marked signal
  probe (`diag.c`) and the `sanity_vanilla` / `sanity_clean` probes (both 138).
- `THREADSAFE=1` (CheriBSD's own build, incl. their prebuilt purecap object,
  dynamic and static): the process hangs.

CheriBSD's base runs `libsqlite3` purecap (via `kyua`), so it is achievable in
their full build system, but porting upstream SQLite to run purecap standalone is
a separate project, orthogonal to this task.

The CHERI verdict depends only on each defect's memory-lifecycle events (which
handle is freed/reused/null when, and whether the offending access dereferences
that memory), so each corpus `before.c` is compiled **VERBATIM** against a
minimal SQLite-lifecycle harness (`mock-sqlite/`) reproducing exactly those
events (open/prepare allocate; close/close_v2 free the connection; finalize frees
the stmt — a 2nd finalize is a real double-free; exec invokes the registered
progress/UDF/authorizer callback; step/reset deref the possibly-freed connection;
column_name returns a stmt-owned buffer freed by finalize). The harness runs
clean purecap (`sanity_mock` exits 0). Shims built **`-O0`** so the UB dangling
access is actually emitted (at `-O1+` the compiler hoists the handle load before
the free or elides the access — verified: at `-O1` several rows spuriously
mismatched; `-O0` made them consistent).

## Results

See `tests/cheri-baseline/RESULTS.md` for the filled 15-row table + the
`row3_reuse` supplementary. Headline (confirms the task's thesis):

1. **Spatial-only CHERI is blind to the whole temporal class** — every UAF /
   use-after-close / reuse / double-free row runs to completion; only the 3
   null-derefs trap.
2. **Realistic async revocation still does not catch UAF at the contract point** —
   the short reproducers free-and-reuse before any sweep; the dangling cap is
   revoked only by a later stop-the-world sweep, never synchronously at
   `step`/`reset`/`finalize`/`close`. Only the double-free is caught here (the
   allocator aborts on the 2nd free, independent of the sweep).
3. **Even revoke-on-every-free catches these only post-free, and CANNOT catch the
   reuse-not-free case at all** (`row3_reuse`, the real diesel pattern): the
   buffer is overwritten in place, never freed, so the capability stays tagged and
   in-bounds — the clean "CHERI can't". Our system faults synchronously at the
   contract point in every row.

## Scope / discipline

Additive, measurement only. The CHERI toolchain lives entirely outside the llvm
tree (`~/cheri-ws`, not committed). **No `llvm/` change, no `capstone-qemu`
change, no gitlink bump.** Only new files under `capstone/tests/cheri-baseline/`.
A's paper/repros/`start.S`/monitor/`capstone-c` untouched; the corpus `before.c`
sources are compiled unmodified. No *our-QEMU* rootfs lock (CHERI has its own
QEMU). No `Co-Authored-By`; no debug/report files committed.
