# Results

28 defects in SQLite 3.22.0, each measured on three arms. The corpus held 33
until the case-by-case review of 2026-10-15/17 withdrew five and renumbered the
rest; `../README.md` carries the mapping and the reasons.

Two configurations of the same 28 defects, one build flag apart. In the nested
one SQLite allocates through **memsys5** over a static array, so every object is
a sub-allocation inside one arena. In the non-nested one `SQLITE_CONFIG_HEAP` is
simply not called and SQLite allocates through the platform.

| configuration | arm | detected | temporal | spatial |
|---|---|---:|---:|---:|
| nested | `virtual-malloc` (base) | **0 / 28** | 0/20 | 0/8 |
| nested | `virtual-nested-pools` (Sublet) | **28 / 28** | 20/20 | 8/8 |
| nested | `cheribsd-revocation` | **0 / 28** | 0/20 | 0/8 |
| non-nested | `virtual-sysmalloc` (base = Sublet) | **24 / 26** | 18/18 | 6/8 |
| non-nested | `cheribsd-revocation` | **24 / 26** | 18/18 | 6/8 |

The rows are not summed: they are the same defects twice. Cases 05 and 18
configure the arena themselves to force an OOM path and have no non-nested form,
which is why that half is 26.

**Nesting is the whole of the gap.** The same mechanisms that see nothing inside
memsys5 catch almost everything when the object is one the system allocator
handed out. They are not weak detectors; they are detectors pointed at a lifetime
the allocator never told them about.

**A fault is not a catch unless its CAUSE is the mechanism's.** The base arm's
apparent nested catches were all `cause 24`, an untagged dereference -- the
architecture's baseline tag check firing on a value the program had just read out
of the dead object, past an access the mechanism had already allowed. Sublet's 28
are `cause 25` (lifetime) or `cause 28` (bounds) at the case's declared site. The
two SQLite configurations therefore declare `catch_causes: [25, 28]` in
`tools/arms.json`, and `tools/verdicts.py` does not score a fault outside that set
as a catch.

The two spatial cases missed in the non-nested configuration, 22 and 23, are
over-reads of exactly one byte past an 8- and a 16-byte object. Host ASan reports
them as "0 bytes after"; no allocator whose granule is coarser than the request
can see that byte, and every allocator's is.

## What produced these numbers

| arm | runner | platform |
|---|---|---|
| `virtual-malloc`, `virtual-nested-pools` | `shared/run-virtual.py`, built by `ports/sqlite/repro322/build-virtual.py` | virtual Capstone, musl mallocng |
| `virtual-sysmalloc` | the same, built without `--memsys5` | virtual Capstone, no nested allocator |
| `cheribsd-revocation` | `shared/run-cheribsd.py`, which invokes `ports/sqlite/cheribsd/build-corpus-cheri.sh` | CheriBSD 15.0-CURRENT riscv64-purecap, QEMU |

Bundles: `2026-10-17-renumbered/<arm>`, derived into each `case.json` by
`tools/derive-verdicts.py`. The `arms` blocks are not hand-edited.

The case list and the case sources come from this corpus, on every arm. Until
2026-10-17 the CheriBSD arm compiled its own copies under
`ports/sqlite/cheribsd/cases/`, and three had drifted into different programs;
`build-corpus-cheri.sh` now takes the program from the corpus directory
`tags.tsv` names and records what it compiled in `out/sources-<group>.tsv`.

Every run is gated. The virtual arms run `controls/control_{uaf,bounds}_mem5.c`,
which must complete on the stock arm and fault on the port; the non-nested arm
runs `control_{uaf,bounds}_malloc.c`, which must fault. CheriBSD runs
`poscontrol.c`, a use-after-free through the system allocator, which must fault or
the run is void.

## The arena is a property of the case

memsys5 is configured over `static unsigned char sqlite_heap[arena_bytes]`, and on
an arena-granular arm that array IS the bound: an over-read is visible exactly when
it leaves it. The three builders disagreed about its size until 2026-10-17 -- the
virtual arms used a 1 MiB header default against CheriBSD's 262144 -- so the spatial
row was comparing arenas rather than mechanisms. Case 26 shows the size deciding: from
one source it is caught on base and on CheriBSD at 262144 and missed on both at
1048576. `arena_bytes` in `case.json` is now read by all three builders.
