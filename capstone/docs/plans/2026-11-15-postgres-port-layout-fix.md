# PostgreSQL memory-context port: stock layout and no block reference in the chunk header, pre-registration, 2026-11-15

Committed before any measured run. A reading that differs from what is written here is a finding;
this file is not edited afterwards.

## Problem

Run-5 (`2026-11-15-postgres-established-workload.md`, records in
`ports/postgres/app/results/2026-11-15-pgbench-reuse/`) found that 7.8 % of the protected arm's
reuses fall in another bucket than stock's (reusediff 0.0777 rep 1 against rep 1), against a stock
noise floor of 0.00005. The cause is the lifetime patch
`ports/postgres/memory-contexts/patches/postgresql-17.0-0003-memory-contexts-sublet-lifetimes.patch`,
not the processor design:

- It adds `void *sublet_parent` (16 bytes) to every `MemoryChunk` header (16 to 32 bytes) and a
  `SubletBlock {generation, allocation, size}` (48 bytes) to every block header of AllocSet,
  Generation, Slab and Bump. Fewer chunks fit per block, so PostgreSQL requests more blocks. The
  T = 1000 census (`diagnostics/census-and-variants.txt`) counts 12,001 MessageContext blocks
  against 10,000 in stock and 177 "index info" blocks against 117. A stock build with only the
  larger headers reproduces the shift (stock against layout 0.0729, protected against layout
  0.0004).
- `MemoryChunkIssue` stores the block's malloc reference in `sublet_parent` and derives the child
  over the header and the payload. A live caller can load the block's reference from the 16 bytes
  in front of its chunk and reach every other chunk of the block. The patch states this itself.

## Change

1. `MemoryChunk` and every block header return to their stock layout: no `sublet_parent`, no
   `SubletBlock` in a block header.
2. Each block's malloc reference, generation and extent live in a block table in `mcxt.c`. The
   table is static memory outside every block: 32,768 block records (64 bytes each, 2 MiB) and a
   two-level page map over the Sv39 user address space (2 MiB leaves of 512 page entries, 8 bytes
   each, at most 1,024 leaves, 4 MiB, and a 512 KiB leaf index). Entering a block calls no
   allocator, so no malloc call interleaves with the managers' block requests. Each 4 KiB page
   names the block that covers its first byte and chains the blocks that start inside it. A lookup
   costs one leaf-index load, one page-entry load and a walk over the blocks that start in that
   page. Entering a block writes one page entry per page it spans.
3. Every manager enters a block right after malloc returns it (the keeper block at context
   creation, new blocks, the dedicated blocks of large chunks, the AllocSet realloc's new block)
   and removes it right before `free`. A full table fails like malloc: the manager frees the block
   and reports out of memory.
4. The child covers the payload only: `MAXALIGN(size)` bytes from `MemoryChunkGetPointer`, one
   byte for an empty request (CDERIVE takes no empty child). Below 4 KiB the bounds are exact, so
   the header lies outside them. Above, the bounds format adds granule slack, as before.
5. pfree, repalloc, GetMemoryChunkContext and GetMemoryChunkSpace first load one byte through the
   caller's pointer (a stale pointer faults with cause 25, as before), then find the block by the
   pointer's address and read the header through the block's malloc reference.
   `GetMemoryChunkMethodID` and `GetMemoryChunkHeader` read the header word the same way, so the
   reuse hooks and the dispatch need no change of text. The allocator lends the block's reference
   only to a pointer at the start of its chunk's bounds. An aligned redirect chunk (its header lies
   inside the caller's bounds), an address in no block and a pointer into the middle of a chunk
   keep the caller's own authority. repalloc refuses such a pointer with an error.
6. Unchanged: generations derived on first use, reset revokes the keeper block's generation, every
   realloc creates the new chunk before it ends the old, the AllocSet single-chunk realloc keeps
   malloc, issue, copy and free (failure-atomic, as the lifetime-failures unit test requires), a
   Bump request for zero bytes keeps its one MAXALIGN unit, and the managers keep writing their
   bookkeeping (free-list links, block headers) through their own references.
7. `hooks-protected.patch` is regenerated with `make-patches.py` from the new protected `mcxt.c`:
   its anchors and hook text are unchanged, only its context lines move.

## Checks

**Unit test** (`ports/common/tests/lifetime-failures/pg-test.c`, built by that directory's
`build.py` with `--baseline 5060011844e`, so its unfixed image carries the run-5 patch; run by
`ports/postgres/memory-contexts/tests/layout-fix/run.sh` in one trap-route boot):

- `header`: a live caller loads the 16-byte words in front of a 64-byte AllocSet chunk.
  Negative control (run-5 patch): the load at -32 returns a tagged reference whose bounds cover
  the chunk's whole block. New patch: the first load (-16) faults with cause 28.
- `stale`: pfree, then a load through the old pointer faults with cause 25, in both images.
- `table`: every header reader through the block table on all four managers (pfree,
  GetMemoryChunkContext, GetMemoryChunkSpace, in-place and moving repalloc, aligned chunks, empty
  and large requests up to 1 MB, resets, more AllocSet contexts than the context freelist keeps).
  Passes on both images.
- The directory's existing modes (CDERIVE fault injection, large sizes, failure-atomic large
  realloc) pass on the new image.

**Static layout**: clang's record layout dump (`-Xclang -fdump-record-layouts`, the build's own
compiler and flags) of aset.c, generation.c, slab.c and bump.c for the stock tree and the new
protected tree. `MemoryChunk`, `AllocBlockData`, `AllocSetContext`, `GenerationBlock`,
`GenerationContext`, `SlabBlock`, `SlabContext`, `BumpBlock` and `BumpContext` must have the same
size and field offsets in both. The run-5 protected tree is the control: it differs.

**Dynamic layout**: run-5's block census (the diagnostic patch that counts every block request per
manager path and context name) applied to stock and to the new protected source, on the T = 1000
calibration workload with `track_counts=off`, as in run-5's census runs. Every row (path, context,
blocks, bytes) must equal stock's, including MessageContext 10,000 blocks and "index info" 117.
The same census build also counts the chunks whose bounds start more than 16 bytes below their
payload (granule slack that reaches past the chunk's own header), as a measured residual.

## Measurement

Images from a tree export of `fix/postgres-port-layout` (the commit that adds this patch), built
with `experiments/study/reuse-hooks/postgres/build.sh` for both arms on p13 under
`~/asr-work/fix-postgres/`. The run-5 stock repetitions are reused only if the new stock image is
byte-identical to run-5's (sha256 43f0d5d3...). Otherwise stock is rerun too.

Exactly run-5's workload and settings: `run-pgbench-reuse.py`, kit `~/asr-work/eval/kit-run-3`
(platform of the brief), workload `bench-postgres-perl/gen/s1-t4000` (tpcb.sql f095983b...,
cluster fa14e868...), share `prebuilt/reuse/pg-share`, step timeout 3600 s, no extra settings.
Two repetitions per arm, one boot each, results in `~/asr-work/eval/run-6/reuse/postgres-ARM-REP/`
with `provenance.txt`, collected with `reuse-hooks/cpython/collect-reuse.py`. The same rejection
rules as run-5: incomplete run, nonzero exit, fault, nonzero observer error, or result rows that
differ from `native.out`. Wall time is recorded for every run.

**Corpus**: the arms that build with patch 0003. In run-4 that is the sql-repros
`virtual-pg-pools` arm (`bug-corpora/postgres/sql-repros/shared/build-trap.sh protected`);
`virtual-malloc` builds without it and is not rerun. The new image runs `run-arm.py` with run-4's
kit (`kit-run-4`), fixture (`prebuilt/reuse/pg-fixture/cluster`, tree 98f7ed38...) and share, in
two boots as run-4's readings came: all cases but 10 with the default step watchdog, and cases 10
and 11 with `CAPSTONE_STEP_TIMEOUT=5400`. Results in
`~/asr-work/eval/run-6/corpora/postgres-sql-virtual-pg-pools/` and
`~/asr-work/eval/run-6/corpora/postgres-sql-10-11-virtual-pg-pools/`. Every case is compared with
run-4, and every changed verdict is explained.

## Predictions

1. **Unit test**: as listed under Checks; the negative control shows the tagged block reference,
   the new image faults with cause 28 at the first header load, stale accesses fault with cause 25
   in both.
2. **Static layout**: identical sizes and offsets in the nine structures.
3. **Dynamic layout**: every census row equal to stock's.
4. **Reuse**: protected-1 against protected-2 at most 0.0001. Stock against protected inside
   the stock noise floor: rep 1 against rep 1 and every other pair at most 0.0001, the order of
   run-5's stock floor (0.00005), against 0.0777 for the run-5 patch. Issues and releases differ
   from stock by no more than the two stock repetitions differ from each other (run-5: 126).
   Median bucket equal. Every run's result rows equal the native replay.
5. **Cost**: the protected arm's guest seconds within 15 % of run-5's protected arm (1,396 s).
6. **Corpus**: every verdict at least as in run-4: cases 01, 02, 03, 04, 06, 07, 08 and 10
   CAUGHT (8 of 11), 05 and 11 MISSED, 09 NO-READING (unattributed), and the four controls
   fault as declared.
