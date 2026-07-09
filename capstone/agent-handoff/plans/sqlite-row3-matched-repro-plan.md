# Plan — row3 matched before→after against real SQLite (close the "abstract probe" gap)

*Status: PROPOSED (A-lane, `capstone-bootstrap`), 2026-07-09. Awaiting go-ahead
before the build/integration step. This is the paper's central empirical claim,
so it is written up for review first (propose-before-big-directions).* 

## Why

The corpus today has a real→abstract **mismatch** the paper cannot lean on as a
matched pair:

- **Before** (`cve-repros/row3_diesel_colname_cached/before.c`, 19 LOC): a real
  SQLite C-API program — `open/exec/prepare/step/column_name/finalize/close`,
  then reads `name[0]` *after* `sqlite3_finalize`. Genuine UAF; host oracle =
  `heap-use-after-free` (ASan).
- **After** (current): `tests/runtime-qemu/sqlite-borrow-revoke-probe` — a
  *hand-written abstract probe* with a magic sentinel (`0x0FA017ED`) that models
  the borrow/revoke mechanism and revokes at `sqlite3_step`. It is **not the same
  program**, does **not** link SQLite, and uses a magic value.

All three validated shapes (BORROW-REVOKE r3/13/18/19, HIERARCHICAL r4/5/7/8/9/10/12,
SEALED r1/2/6/16) point at three such abstract probes. None is a matched
real-SQLite before→after. `stage2-mapping.md` already notes the fix for r3: *"move
the validated probe's revoke from `sqlite3_step` to statement finalization."*

## What already exists (verified 2026-07-09 — do NOT rebuild from scratch)

- **Real SQLite compiles under Capstone.** `benchmarks/sqlite/build-sqlite-capstone.sh`
  fetches the official amalgamation (SQLite **3.53.3**, SHA3-verified via
  `fetch-sqlite.sh`), patches only the `SQLITE_TRANSIENT` function-pointer sentinel
  (the 128-bit-cap constant evaluator asserts on `(destructor_type)-1`), and links
  it with `adapted/capstone_sqlite_{os,libc}.c` + `runtime-qemu/sqlite-vfs-skeleton`
  into a domain `sqlite_memory_capstone.dom`.
- **It runs real queries in QEMU.** `run-sqlite-memory.sh` boots the domain with a
  host driver (`sqlite_host.c`) and asserts real result rows
  (`row name=alpha value=11` … `__CAPSTONE_SQLITE_MEMORY_PASSED__`).

So the matched after is an **integration on a working harness**, not new infra.

## Deliverable

The **same** row3 program (real SQLite API, column-name cached across finalize),
compiled into the real-SQLite domain, where the domain mints a revocation
capability for the column-name buffer and **revokes it at `sqlite3_finalize`**, so
the post-finalize `name[0]` read **traps with a real capability fault** — no magic
sentinel. Result: one program, two outcomes on identical source —
host = ASan heap-use-after-free, Capstone = deterministic cap fault. That is the
matched pair the paper needs.

## Steps

1. **Reproduce the host UAF** for row3 against real SQLite 3.53.3 (build
   `before.c` with ASan via `build-sqlite-host.sh` path) — confirm the oracle
   line. No Capstone build. (Grounds the "before".)
2. **Design the revoke hook (the one real design choice).** Decide where the
   domain mints/revokes the column-name borrow *without* patching the 250k-LOC
   amalgamation: candidate = a thin domain-side wrapper around
   `sqlite3_column_name`/`sqlite3_finalize` in the host-call glue
   (`sqlite_hostcall.h` / `sqlite_capstone_domain.c`), minting an R-cap for the
   returned buffer at `column_name` and revoking it in `finalize`. Write this into
   the plan and get sign-off — it is the analogue of the abstract probe's
   revoke, moved onto the real call path and to the finalize point.
3. **Wire it** into the domain build; drive the row3 sequence from the host
   driver (extend `sqlite_host.c` or a row3-specific driver).
4. **Show the trap.** Post-finalize `name[0]` read faults; capture the cap-fault
   halt line as the "after" oracle (replacing the sentinel-based probe reference
   in the row3 README).
5. **Regression:** the existing `run-sqlite-memory.sh` markers still pass (the
   revoke wrapper must not perturb the normal query path).
6. **Fold the honesty fix** into `stage2-mapping.md`: mark r3 as *matched
   real-SQLite before→after* and keep the abstract probes labelled as
   *mechanism probes*, so the paper's claim table is precise.

## Scope / cost

- Needs the Capstone clang build (already present in this clone) + one QEMU domain
  run (serialize per the single-`rootfs.ext2` rule — A-lane owns firmware/QEMU, no
  contention with B's compiler lane).
- ~1 focused build/integration pass once step 2 is signed off. Rows 19/13/18
  (same BORROW-REVOKE template) become cheap follow-ons afterward.

## Open question for review

Step 2's hook location: thin host-call wrapper (recommended — no amalgamation
patch, revoke lands exactly at the API contract point) vs. a VFS/pager-level
interception. Confirm the wrapper approach before I build.
