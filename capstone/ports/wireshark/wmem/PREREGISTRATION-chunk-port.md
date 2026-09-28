# A per-chunk Sublet port of wmem's block allocators — pre-registration

Written and pushed **before the first ported line**, because `capstone/tests/port-effort.py` states
the rule it enforces: the counting rule "has to stand before the first ported line and must never
be adjusted at the result, because a number produced after the fact cannot be checked."

## What exists, and what this adds

`patches/wireshark-4.6.8-0001-wmem-authority-hooks.patch` (25 hunks, 142 changed lines) gives wmem's
four allocators authority at **region granularity**: one region per `wmem_alloc(NULL, n)`, with
narrow / widen / epoch hooks. Its README names the limit plainly:

> an individual `wmem_free` in the `block` allocator returns the chunk to a free list inside a live
> block … a chunk has no region of its own, so this free ends no epoch (fixture 4).

So a chunk free is not covered. The same port's PoisonCap backing *does* invalidate at "reset,
release and chunk free", so as things stand **PoisonCap covers a lifetime Sublet does not** — not
because the mechanism cannot, but because nothing carves a chunk into a region. memsys5 shows the
alternative, and this port is that: carve each chunk as its own sub-region so a chunk free becomes a
revoke.

That existing port has a `ledger.manifest` marked "inventory only, not a measured effort claim" and
**no `.classes`**, so it is not in the A7 count. This one will be.

## The classification rule, fixed now

One line per hunk in `<patch>.classes`: `file:new_start  class  level  note`, classes
`hierarchy|metadata|workaround`, levels `allocator|application`. Decided in advance for this patch,
so no hunk's class is chosen once its effect on the total is visible:

- **hierarchy** — taking the block linear from the level below, the per-chunk `sublet_split` and
  `sublet_take`, the `sublet_give` on a chunk free, the senior `sublet_give_to` that a `free_all`
  or a destroy performs, and any signature change that carries a region or an index instead of a
  raw pointer.
- **metadata** — anything moved out of memory that is handed away or revoked: the free-list links,
  the recycler, the chunk headers and the block list, and the side tables that replace them.
- **workaround** — present only because a compiler or RTL defect demands it. I expect **zero** here;
  a non-zero count is a finding and its note must name the defect.
- **level** is `allocator` for `wmem_allocator_block*.c`, and `application` for anything above
  wmem's interface. **H1 predicts zero application lines.**

**Out of the count, in `ledger.manifest` instead:** patch 0007's block-size change (forced by the
16 MiB pool, not by the discipline) and the shim/glue files, exactly as nginx kept `ngx_shim.h`
outside its own count.

## Predictions

- **P1 — the point of the port. Fixture 4 must FAULT in `sublet` mode** and still complete in
  `spatial`. It currently faults in neither, and its comment calls that a documented limit. This is
  a control that already exists and whose expected value flips, which is the cleanest before/after
  available.
- **P2 — a side effect on the UNPROTECTED arm, registered because it is easy to mistake for
  breakage.** Fixture 3's comment says the stale read "returns allocator metadata, not the old
  byte", because the free-list node is written into the freed chunk's data. Once the free list moves
  out, that is **no longer true**: the spatial arm should read the old byte, not a node. So
  fixture 3's expected *unprotected* value changes, and its comment must be corrected in the same
  patch. If it does not change, the metadata did not actually move.
- **P3 — the A7 split.** H4 predicts metadata carries most of the lines. wmem's free list lives
  inside freed memory, so H4 should hold here; it was refuted once already by PostgreSQL, so this is
  a real test. **Prediction: metadata > hierarchy in added lines, and workaround = 0.** Recorded
  now; the rule is not adjusted afterwards whatever the number says.
- **P4 — counters.** A `free_all` costs **one revoke per retained block**, not one per chunk. A
  chunk free costs one revoke. Both are measured, because they are the hierarchy claim.
- **P5 — behaviour unchanged.** The replay validator passes across all four allocators, and the
  other lifetimes fixtures keep their current verdicts.

## Scope

`wsutil/wmem/wmem_allocator_block.c` and `wmem_allocator_block_fast.c`. The `simple` and `strict`
allocators already give every object its own region and are unchanged.

**Two steps, and only the first is promised here:** (1) the ported allocators in the existing wmem
replay harness, which has the fixtures and the runner; (2) the same patch compiled into **real
tshark** via `ports/wireshark/app/host/build-domain.sh`, which already recompiles exactly these two
files for the sublet arm. Step 2 is what makes it more than a model, and its own predictions will be
registered separately.

## Risks

- wmem's chunk model is richer than memsys5's — `prev`/`last`/`used`/`jumbo` bits, a recycler, and
  merging of adjacent free chunks — so the metadata move is the bulk of the work.
- **Merging is the hard part:** two adjacent free chunks becoming one requires the senior handle
  discipline (`sublet_handle` before the split, one `sublet_give_to` to rejoin), as memsys5 does.
- Jumbo chunks bypass the block path and need their own handling.
- `realloc` in place is where a naive port silently loses protection; it needs its own fixture.

## Addendum, 2026-09-29: v2 and its checks, registered before v2's first build

The first run (v1, commit `e5d20a1b27d3`) met P1-P5, and a claim audit then found gaps in what that
run could show. Recorded here before v2 is built, with every prediction v2's run is judged by.

**v2 changes the allocator in one respect.** v1 kept upstream's free-list minimum -- room for two
pointers in the chunk's data, 32 bytes in purecap -- although the links had moved to the record. A
freed chunk below it joined no list, and since chunks never rejoin, it was lost until the block's
next reset. v2 lists every chunk that can serve a request (`WMEM_FREE_MIN_LEN`), and a shrink no
longer splits off a remainder too small to list. The classification rule is unchanged: the new
hunks are the recycler, which the rule names as metadata.

**New fixtures, 13-17** (`security-tests/shared/lifetimes.c`), on the port (v2):

| case | what | mode 0 | mode 1 |
|---|---|---|---|
| 13 | read through the old pointer after a GROWING realloc | completes | FAULT at `wm_probe_read` |
| 14 | the same after a SHRINKING realloc that splits (same address) | completes | FAULT at `wm_probe_read` |
| 15 | read after freeing a BLOCK-allocator jumbo | completes | FAULT at `wm_probe_read` |
| 16 | a second free of a chunk inside a live block | completes (not run) | FAULT at `wm_widen_probe` |
| 17 | 1000 alloc/free of 8 bytes reuse ONE address | completes | completes |

Cases 0-12 keep v1's verdicts: 36 cells, all PASS.

**Matched arms**, each one variable away from the port:

- `WM_CHUNKS=OFF` (the hooks alone, same tree), mode 1: 13 **completes** (upstream grows in place,
  so the old pointer is the live one); 14 **completes**; 15 **faults** at `wm_probe_read` too (a
  jumbo is a region under the hooks as well -- registered as NOT discriminating); 16 does **not**
  fault at `wm_widen_probe` (otherwise unconstrained: the free lists may be corrupted); 17
  completes. Mode 0: 13, 14, 15 and 17 complete.
- `WM_P1_ABLATE` (the port, with the one `sublet_give` of `wm_chunk_retire` stubbed out), mode 1:
  cases 4 and 13 **complete**. That attributes their faults to that revoke alone, which v1's
  control -- a whole patch away -- could not.
- v1's allocator swapped into the same build (the generated `wmem_allocator_block.c` replaced by
  v1's port output, with fixtures and everything else unchanged): case 17 **fails with status 528**
  in both modes. This is the positive control for the reuse check.
- `WM_P2_CONTROL`, re-run from the committed tree: m0c3 status 519 and m0c4 status 520, as in v1.

**Counters.** The report now carries both translation units' `sublet_stats`, because `sublet.h`
counts per translation unit. The audit showed that v1's `revokes=140` was the chunk port's unit
alone. Replay, protected mode: `resets=40 closes=20 reset_revokes=40 close_revokes=20`. The chunk
unit's revokes equal `reset_revokes + close_revokes + retires`, which is the "no hidden revoke on a
reset" claim, now stated as an equality. `region_revokes` and both `inits` are reported, not
predicted: there is no prior count for them. The spatial replay reports 0 revokes and 0 inits in
both units.

**Behaviour.** Both QEMU replay reports equal 2026-09-21's field for field, and the native replay
still matches unmodified upstream. The audit showed that this equality cannot see the port's
placement changes, because the checksum is a function of the trace. So it is claimed only as what
it is: the same trace completes, with the payload checks at every free, realloc and reset passing.
The split-overlap mutation that shows those checks fire is re-run, and its output is kept.
