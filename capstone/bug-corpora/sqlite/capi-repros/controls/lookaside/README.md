# Lifetime controls at SQLite's lookaside

Not defects. These are the three controls of
[`docs/plans/2026-10-11-nested-lifetime-controls.md`](../../../../../docs/plans/2026-10-11-nested-lifetime-controls.md),
applied to the allocator that
[`docs/plans/2026-10-12-nested-lifetime-controls-extended.md`](../../../../../docs/plans/2026-10-12-nested-lifetime-controls-extended.md)
(5d336c73931c) adds for SQLite: the per-connection **lookaside**. The memsys5 controls in `..`
cannot reach the lookaside, because `sqlite3_malloc` never uses it.

Each control opens a `:memory:` connection with the default lookaside (`SQLITE_DEFAULT_LOOKASIDE`
1200,100). It takes blocks with `sqlite3DbMallocRaw(db, 128)` and releases them with
`sqlite3DbFree(db, p)`. These are the two internal entry points that the lookaside half of
`ports/sqlite/sublet/sublet-3220000-memsys5.patch` instruments:
- `lookasideIssue` derives each slot as a CDERIVE child of `lookaside.pLife`;
- `lookasideRelease` revokes it with CREVOKE, then reaches the slot through `lookaside.pStart`.

The heap under the lookaside is the memsys5 heap that the memsys5 controls use.

| control | image | what it does | expected, both arms |
|---|---|---|---|
| `control-parent` | `la-control-parent.dom` | takes A and D and releases D, then A. It then reads and writes A's former range through `db->lookaside.pStart + offset`, the reference the lookaside keeps to its buffer | completes. The read returns the lookaside's own `LookasideSlot.pNext` = D (the previous free-list head), followed by A's bytes. The write is undone before the lookaside runs again, and the next two requests get A and then D |
| `control-sibling` | `la-control-sibling.dom` | takes A and B, releases A, then reads B, writes B and reads B back | completes; B intact |
| `control-successor` | `la-control-successor.dom` | releases A, then takes C of A's size until the lookaside returns A's start (at most 64 requests, each kept), then writes and reads C | completes; C's start equals A's |
| | `la-control-successor-stale.dom` | the same sequence, then one read through the kept reference to A at the probe `stale-A-read`. It is a separate image, so it runs in a separate boot | protected: fault, cause 25, at that probe. Release: completes, and the read returns C's bytes |

## A is a lookaside slot

`la_alloc()` (`la_common.h`) is the only way a control takes a block. It prints a `SLOT` line, and
the `CHECK <name>-is-lookaside-slot` check is `ok` only when all four of these hold:
- the address is in `[db->lookaside.pStart, pEnd)`;
- its offset from `pStart` is a multiple of `lookaside.sz`;
- the lookaside hit counter `anStat[0]` went up by exactly 1 across the request;
- `sqlite3DbMallocSize(db, p)` returns `lookaside.sz`.

`slot-check-negative/` is the positive control for that check. Two requests on the same connection
must fail it, and fail it on both arms:
- a request larger than a slot;
- a request with the lookaside disabled.

That program is not part of the pre-registered set. `slot-check-negative.py` builds it with the
exact command `build-controls.sh` recorded, and boots it.

## Probes

The byte probes are `PROBE_READ` / `PROBE_WRITE` from `../control_common.h`.

The lookaside's free-list link is a pointer, and a byte copy drops its capability tag. So the
parent control reads the link and restores it with `PROBE_LOAD_PTR` / `PROBE_STORE_PTR`
(`la_common.h`). Each of these is one typed access on one source line, and each prints the same
`PROBE … file:line` / `PROBE … done` pair.

## Build and run

    build-controls.sh <virtual SDK> <amalgamation dir> <out>/release   release
    build-controls.sh <virtual SDK> <amalgamation dir> <out>/protected protected
    run-controls.py --builds <out> --kit <platform kit> --llvm-bin <dir> --work <new dir>
    slot-check-negative.py --builds <out> --kit <platform kit> --llvm-bin <dir> --work <new dir>

`build-controls.sh` and `run-controls.py` are copies of the memsys5 ones in `..`. Only these differ:
- the image names (`la-control-*`);
- the repository depth;
- the include path (`..` for `control_common.h`);
- the address lines the runner keeps.

Staging, flags, arms and the judge are unchanged.

Results: [`../../results/2026-10-12-controls-ext/`](../../results/2026-10-12-controls-ext/).
