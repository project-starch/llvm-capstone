# A lookaside slot is reused while memsys5 still owns the block

The first W3 quantity, measured. It is **not** the W3 row for
`sqlite-lookaside`, because it was recorded on SQLite 3.42.0, the revision
the existing survey pins, and not on 3.53.3, the revision the port pins.
What it establishes is that the nested question is answerable, what the
answer is on this workload, and that the pinned instrument could not have
answered it.

## What was asked

`levels.json` gives `sqlite-lookaside` the level `lookaside` with
`memsys5` beneath it rather than `libc`. The question is whether a
lookaside slot is handed out again before the allocator *directly beneath
it* has taken the block back. The existing survey cannot ask that: it
asks every level about the system allocator.

## Why the pinned instrument cannot answer it

`memhook.c` asks its reuse question at lookaside and answers it against a
stamp arena that only `hk_free` and `hk_realloc` write. Both wrap
SQLite's `sqlite3_mem_methods`, so the clock is the system allocator's.
The instrument says so in its own header: with `--heap`, speedtest1
replaces that table by memsys5 and level 0 "sees nothing, which is the
truth: nothing below lookaside reaches the system allocator then".

The consequence is visible in the pinned arm's own numbers, which are
reproduced here as `memsys5.pinned.rep*.hook.txt`:

    @reused_before_return:          1876536
    @reused_block_never_returned:   1876536

The two are equal, which means no stamp was ever written and the clock
never ran. A reading of "reuse precedes release" from that arm would be
reading an idle counter.

## The change

`patches/memhook-stamp-level-beneath.patch`, four lines: stamp the
release clock at `a1_memsys5_free` as well, so it follows whichever level
sits beneath lookaside. In an arm where memsys5 is idle nothing changes.

## The result

Three repetitions, `memsys5.nested.rep*.hook.txt`. The three report
files are **byte-identical**, which is a property of this workload and
not a copied file: speedtest1 at a fixed size over an in-memory
database performs the same allocation sequence every time, and the
existing survey's five repetitions of the same arm agree the same way.
Their shared hash is in `raw/SHA256SUMS`.

| quantity | value |
|---|---|
| lookaside allocations | 1,876,673 |
| lookaside reuses | 1,876,536 |
| reuses before the block returned to memsys5 | **1,876,536**, all of them |
| reuses after it returned | 0 |
| distinct addresses lookaside handed out | 137 |
| most objects at one address | 280,102 |
| memsys5 allocations / releases | 652,201 / 652,201 |

The stamped number alone would be ambiguous, because "no stamp" and "a
stamp that is too late" both read as before-release. `diag2.hook.txt` is
a throwaway build, `patches/memhook-diagnostic.patch`, that separates
them by recording when each lookaside address was first covered by a
memsys5 release:

    @stamp_calls:                        652201
    @stamp_cells:                      80721880
    @lookaside_addresses_seen:              137
    @lookaside_addresses_ever_stamped:      137
    @lookaside_first_stamp_at_alloc:    1876673
    @lookaside_last_stamp_at_alloc:     1876673
    @lookaside_total_allocs:            1876673

Every memsys5 release stamped, eighty million cells were written, and
every one of the 137 lookaside addresses was covered **exactly once, at
allocation 1,876,673**, which is the last one. The block holding the
lookaside buffer is returned to memsys5 once, at teardown, after the
final reuse.

## What it says

Every one of the 1,876,536 slot reuses happens inside a block that the
allocator directly beneath has not taken back. The existing survey row
says these reuses are invisible to the system allocator. This says they
are invisible to memsys5 as well, which is a level the system allocator
cannot see either, and it is the statement a single row per program
cannot make.

## What it does not say

It is 3.42.0, not the ported 3.53.3, so it does not complete the W3 row.
It is one workload, speedtest1 at size 100. A release through
`a1_memsys5_realloc` is still unstamped because that entry point receives
no pointer, which is 15,807 of 652,201 memsys5 releases, 2.4 percent, and
every one of them could only move a reuse from before to after, so the
reported share is a lower bound rather than an estimate.

## Provenance

| | |
|---|---|
| SQLite | 3.42.0 amalgamation, fetched and checksummed by `experiments/a1/Makefile` |
| workload | `--heap 268435456 64 --memdb --size 100 --verify --stats` |
| instrument | `memhook.c` at llvm-capstone `72f580c55cd2d245a842c00105ba7c337321e248`, sha256 `e8c7a752e1509336ea91c9627ac4ab62b8fe0bcd0e7bbd2fc5fa63e21762b431` |
| instrument as run | plus `memhook-stamp-level-beneath.patch`, sha256 `70f6178253722a3d01992192fa8e2b9793582662e5f4f9d34ee57cfb8d41389b` |
| hook patch | `experiments/a1/x86/hook-3420000.patch`, applied with `-F0` |
| compiler | gcc 13.3.0 |
| flags | `-O2 -g -DSQLITE_ENABLE_MEMSYS5 -DSQLITE_THREADSAFE=0 -DSQLITE_DEFAULT_MEMSTATUS=1 -DSQLITE_ENABLE_RTREE` |
| raw | `raw/`, hashed in `raw/SHA256SUMS` |

Builds and downloads stayed outside both repositories.
