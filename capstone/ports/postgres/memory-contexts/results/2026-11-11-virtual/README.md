# Memory-context port on the one-operand CREVOKE and CRENEW, 2026-11-11

The port's programs on the virtual platform after the lifetime instructions changed, and what the
conversion of patch 0003 left open. `gate.tsv` holds one line per program run. No consumer defect
is measured here; those are in `bug-corpora/postgres/mmgr-repros/results/2026-11-11-virtual`.

## What ran, and what it showed

| suite | spatial (0001, 0002) | sublet (0001 to 0003) |
|---|---|---|
| native CTest (host cc, the nightly `postgres-mmgr` gate) | 9 of 10: pin, source guards, both replay controls, `contexts-native`, four clients pass | n/a |
| `replay` on the fixture trace and on the mixed-context trace | `PG_REPLAY PASS`, twice | `PG_REPLAY PASS`, twice, same counts |
| four client examples, `allocator-example` | `RESULT 0` four times, exit 0 | the same |
| `contexts-native` | result 0 three times, policy hashes as recorded on 2026-09-18 | fault, cause 25 |

The one native CTest that fails is `port-support`: 53 of its 54 unit tests pass, and
`test_a_rebuilt_sdk_is_invisible_at_head` asks `git log` about a tree that was a plain copy
without `.git`. It does not reach the port.

`contexts-native` faults on the sublet build by the patch's own rule and not by the conversion.
`tests/contexts-workload.h` calls `repalloc(p, 64)`, requires the result to equal `p`, and then
frees through `p`. Patch 0003 ends the caller's reference at every `repalloc` and issues a new
one at the same address, so the old `p` is stale and `pfree` faults reading its header
(`GetMemoryChunkMethodID`, called from `SubletRelease`). The workload is a native test
(`cmake/Tests.cmake` adds it for the native target only). On unmodified dev the same program
faults earlier, with cause 29, on the path described next.

## A block reached from a chunk went back to malloc through the generation

`AllocSetFree` and `AllocSetRealloc` (single-chunk blocks), `GenerationFree` and `SlabFree` find
a chunk's block from the pointer `pfree` hands them, and that pointer is derived from the block's
generation. They then unlink, keep, `free` or `realloc` the block through it.

- **On unmodified dev** (two-operand `CREVOKE`, patch sha256 `8b36b57652ec…`) every one of these
  paths faults in `free` with cause 29: the generation is not a direct child of the mapping
  malloc revokes from. Four loops show it (`palloc(100000)`/`pfree`, the same with `repalloc`, a
  Generation and a Slab context whose blocks empty), and two controls beside them complete (small
  chunks only; contexts created, reset and deleted, where blocks are freed through the lists).
- **As converted** (one-operand `CREVOKE`, patch sha256 `c63524ea5731…`) nothing refuses the
  call any more: `free` ends the lifetime it is given, the generation, and the block's own
  lifetime stays live while malloc reuses the memory. `probe/00` shows it. A 600-byte request in
  a context with a small chunk limit gets a single-chunk block that malloc serves from a group of
  several slots; after `pfree`, a read through the reference malloc returned for that block
  **completes** (`PROBE block reference STILL LIVE after pfree`). On the spatial build the same
  read faults with cause 25, because there `free` got malloc's own reference. `probe/01` frees
  the same block through the context's list (a reset) and faults with cause 25 on all builds.
  With a 100000-byte request the reference dies on every build (cause 25 on all three). That is
  what one expects if malloc gives such a block a mapping of its own and unmaps it at `free`;
  this record did not check that explanation.
- **In this tree** (patch sha256 `29b94c5a026e…`) the block header keeps the pointer malloc
  returned (`SubletBlock.self`) and the four paths take it before they touch the block. `probe/00`
  then faults with cause 25, as on the spatial build.

### The lists a block sits on, and the frees that walk them

`SlabReset` (and `SlabDelete` through it) frees every block through a list link, from the empty
list and from the three block lists. Those links are written in four places: a new block and a
block `SlabAlloc` took from a list carry the pointer malloc returned; the two in `SlabFree` (a
block moved to another list, an empty block put on the empty list) carry `SlabFree`'s own block
pointer. As converted that pointer came from the chunk, so every Slab block that had a chunk
`pfree`'d went back to malloc through its generation at the next reset or delete. `probe/02` and
`probe/03` show it, one per loop of `SlabReset`: a block is filled, one chunk (or every chunk) is
`pfree`'d, the context is reset, and a read through the link taken before the `pfree` **completes**
on the patch as converted. `probe/04` resets without a `pfree` and faults with cause 25 on every
build.

In this tree `SlabFree` replaces its block pointer by `self` before it links anything, so the
lists hold malloc's pointers again and `SlabReset` needs no change: `probe/02` and `probe/03`
fault with cause 25, as on the spatial build. The same holds for the one pointer `GenerationFree`
keeps (`set->freeblock`).

Read against the patched sources, site by site:

| path | pointer that reaches `free` or `realloc` | |
|---|---|---|
| `AllocSetFree`, `AllocSetRealloc` (single-chunk block) | `self`, loaded right after `set = block->aset` | fixed here |
| `GenerationFree`, and `GenerationRealloc` through it, then `GenerationBlockFree` | `self`, loaded before `block->nfree += 1` | fixed here |
| `SlabFree` (block freed at once, or linked) | `self`, loaded right after `slab = block->slab` | fixed here |
| `SlabReset`, `SlabDelete` | the list link, which is malloc's pointer once `SlabFree` links `self` | closed by `SlabFree` |
| `GenerationReset`, `GenerationDelete`, `BumpReset`, `BumpDelete` | the `dlist` link, written only where a block is created | clean as converted |
| `AllocSetReset`, `AllocSetDelete` | `set->blocks` and `next`, written where a block is created or reallocated, and copied from stored links at an unlink | clean as converted |

`CLOBBER_FREED_MEMORY` is outside what the patches support, but the order holds there too: at each
of the four sites `self` is in the local `block` long before `wipe_mem(block, ...)`, and no site
reads `self` from the block after the wipe.

The probes read `AllocSetContext`, `AllocBlockData` and `SlabContext` through their leading fields
(`probe/shared/blockref.h`, `probe/shared/slabref.h`), check what they found against the chunk and
the context's sizes, and build like corpus cases:
`-DPG_CORPUS_DIR=<a directory holding the contents of probe/ and, in shared/, the mmgr corpus's driver.c and corpus.h>`.
All five were built from these files and run on the three builds in one boot.

## Measured about the patch

| | dev | as converted | this tree |
|---|---|---|---|
| patch sha256 | `8b36b57652ec…` | `c63524ea5731…` | `29b94c5a026e…` |
| added / removed lines, hunks | 150 / 12, 37 | 149 / 12, 37 | 158 / 12, 41 |
| capabilities added to a block header | 2 (lifetime, generation) | 1 (generation) | 2 (own pointer, generation) |

Lifetime instructions in the 17.5 server's manager objects at `-O1` (`CDERIVE` / `CREVOKE` /
`CRENEW`, counted by `ports/common/application/lifetime-instructions.py`), the same as converted
and in this tree: `aset.o` 16 / 0 / 1, `mcxt.o` 8 / 1 / 0, `generation.o` 14 / 0 / 2, `slab.o`
6 / 0 / 0, `bump.o` 12 / 0 / 2. The one `CREVOKE` is `SubletRelease`.

## Platform and builds

`capstone-application` preset, `Debug`, on a corpus SDK (`CAPSTONE_SDK_CORPUS=1`) built from this
tree; compiler sha256 `0bbb1f88bc90…`. The virtual kit built from this tree: QEMU `2d5a7107890e…`
(the model with the one-operand `CREVOKE` and `CRENEW`), launcher `0f8e0f903725…`, module
`e47bca62addf…`, pinned mode with the module's defaults, `-cpu` extras
`x-capstone-test-csrs=true,x-capstone-exact-bounds=true`. Every program ran as a process under
`capstone-vexec` in a staged boot (`tools/virtualvm.Batch`). The dev control used dev's own tree,
SDK, launcher, module and QEMU (`0542c5556c`), with 32 supervisor slots.
