# `sqlite-lookaside`, on the revision the port pins

The first W3 row that is complete on its own terms: the level, the level
beneath it, and SQLite 3.53.3, which is what
`capstone/ports/sqlite` builds. The earlier bundle beside this one asked
the same question of 3.42.0 and is kept as the comparison.

## Result

Three repetitions, byte-identical, which is a property of speedtest1 at a
fixed size over an in-memory database rather than a copied file.

| quantity | 3.53.3 | 3.42.0 |
|---|---|---|
| lookaside allocations | 4,550,089 | 1,876,673 |
| lookaside reuses | 4,549,921 | 1,876,536 |
| reuses before the block returned to memsys5 | **4,549,921**, all | 1,876,536, all |
| reuses after it returned | 0 | 0 |
| distinct addresses | 168 | 137 |
| most objects at one address | 281,923 | 280,102 |
| memsys5 allocations / releases | 1,693,135 / 1,693,134 | 652,201 / 652,201 |

The workload roughly doubles between the two releases and the answer does
not move: every reuse of a lookaside slot happens inside a block memsys5
has not taken back.

## Why the number needs the diagnostic

"No stamp was ever written" and "a stamp arrived too late" both read as
before-release, and the pinned instrument produces the first of the two
without saying so. `raw/memsys5.pinned.rep1.hook.txt` is that instrument
on this arm, kept as the control.

`raw/diag.hook.txt`, built with `patches/memhook-diagnostic.patch`,
separates them:

    @stamp_calls:                      1693134
    @stamp_cells:                    293701964
    @lookaside_addresses_seen:             168
    @lookaside_addresses_ever_stamped:     168
    @lookaside_first_stamp_at_alloc:   4550089
    @lookaside_last_stamp_at_alloc:    4550089
    @lookaside_total_allocs:           4550089

Every memsys5 release stamps, 293 million cells are written, and all 168
lookaside addresses are first covered at allocation 4,550,089, the last
one. The block holding the lookaside buffer goes back to memsys5 exactly
once, at teardown, after the final reuse.

## Against the same level over the system allocator

The companion pass in the manuscript,
`experiments/a1/results/x86/20260921T213027Z`, records the same release
with lookaside over libc in its `shipped` arm: 4,489,977 allocations,
4,489,854 reuses, all before backing release, 123 distinct addresses.

The two rows say different things. The `shipped` row says these reuses
are invisible to the system allocator. This row says they are invisible
to memsys5 as well, and memsys5 is itself a level the system allocator
cannot see. One row per program can only make the first statement.

## Provenance

| | |
|---|---|
| SQLite | 3.53.3 amalgamation, pinned in `experiments/a1/sources.sha256` |
| hook patch | `experiments/a1/x86/hook-3530300.patch`, applied with `-F0` |
| workload | `--heap 268435456 64 --memdb --size 100 --verify --stats` |
| instrument | `memhook.c` at llvm-capstone `72f580c55cd2d245a842c00105ba7c337321e248` plus `patches/memhook-stamp-level-beneath.patch` |
| companion pass | `experiments/a1/results/x86/20260921T213027Z`, four arms, five repetitions, cross-checked against the unhooked build |
| compiler | gcc 13.3.0 |
| raw | `raw/`, hashed in `raw/SHA256SUMS` |

## Limits

One workload. A release through `a1_memsys5_realloc` is still unstamped
because that entry point receives no pointer, and every such release
could only move a reuse from before to after, so the share is a lower
bound rather than an estimate.
