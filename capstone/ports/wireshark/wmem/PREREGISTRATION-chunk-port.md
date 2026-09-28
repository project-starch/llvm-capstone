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
