# Protected nested allocators keep their memory behaviour: pre-registration, 2026-10-11

Committed and pushed before any run below. A reading that differs from what is written here is a
finding, recorded as such; this file is not edited afterwards.

## The question

Does protecting a nested allocator with the two lifetime instructions change how it uses memory?
The lifetime patches wrap hand-out and release and leave placement and free-list logic alone, so
the allocator's own decisions should not change. Several patches do add per-object or per-page
metadata, and the processor holds released lifetimes until it reclaims them; those costs are
measured, not assumed away.

Each program runs one workload on two arms from one source snapshot, on the platform of
`2026-10-11-protected-same-answers.md` (QEMU af37cc32, launcher 3975314a, module a1c6cb6b, exact
bounds, one hart): **stock** (nested allocator as released) and **protected** (its lifetime
patch). Images are those of the same-answers runs where they exist. Three repetitions per arm,
run in parallel VMs.

## What is measured, for every program

1. The allocator's own statistics, where the program exposes them (table below).
2. `capstone-vexec --stats` before and after: lifetimes allocated, live, retired and reclaimed,
   collections, and the lifetime table's size.
3. The process's peak resident memory (`VmHWM`), where the guest can read it.

## Predictions

**Decisions are equal** (counts of allocations, frees, objects, pages, contexts and size-class
occupancy, as each allocator counts them): protected equals stock, every repetition.
**Footprint differs only by the documented metadata**, in the direction given; its size is
measured, not predicted. **Lifetimes:** protected allocates more than stock, by about one per
nested-allocator hand-out, and reclaims them.

| Program | Workload | Allocator statistics | Predicted equal | Predicted difference (protected vs stock) |
|---|---|---|---|---|
| SQLite 3.22.0 | speedtest1 `main --size 1`, 8 MiB memsys5 heap | `--stats` / `sqlite3_status`: malloc count, outstanding allocations, lookaside hits and misses, memory used | malloc count, outstanding allocations, lookaside counts | memory-used high-water lower or equal: memsys5 reports the request instead of the block (patch). The pool holds fewer allocation units (4 bytes each for the recorded request). |
| PostgreSQL 17.5 | `work.sql`, single-user | `pg_backend_memory_contexts` at the end | context set (names, levels, count) | none predicted for total bytes: the generation is stored in the existing chunk header |
| CPython 3.13.7 | a fixed allocation workload (the JSON/GC script of `run-ports.py`) | `sys._debugmallocstats()` | arenas, pools and blocks in use per size class | none for pymalloc itself; the radix tree's leaf grows by two fields |
| mruby 4.0.0-rc2 | the AO render benchmark and the GC smoke | `ObjectSpace.count_objects`, `GC.stat` | object counts per type, heap pages, GC runs | heap memory higher: each heap page gains an `issued[]` array of one capability per slot |
| memcached 1.6.45 | the protocol oracle's set/get workload | `stats`, `stats slabs` | items stored and retrieved | per-class chunk counts may move between classes: every item grows by one capability, and cache.c objects by one capability in front |
| Perl 5.36.3 | `smoke.pl` at 2,000 objects | none exposed | — | `VmHWM` higher by the arena trailer, one capability per SV head |
| FFmpeg 9.0.1 | the decode workload | none exposed | — | small; pool entries gain two pointers |
| Wireshark 4.6.8 | the capture set | none exposed | — | small; each block header gains two lifetimes |

Where a statistic is not exposed or `VmHWM` is unreadable, the cell is "not measured", never
zero. N = 3 per arm.
