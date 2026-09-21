# `9e0b4b1ab5` — a change record's LSN read after it went back to its Slab

The corpus's deterministic row: Slab returns the identical address on every
free-then-allocate, so the reuse this defect depends on is a property of the
allocator rather than an accident of a run.

    bash ../run-host-repros.sh 9e0b4b1ab5_reorderbuffer_spec_insert_lsn

## The defect

An `INSERT ... ON CONFLICT` reaches logical decoding as a delayed speculative
insert. When its confirmation arrives, `ReorderBufferProcessTXN` aliases the loop
cursor onto the delayed record, then hands that record back to its context on the
way out — and every hundredth change reports progress by reading `change->lsn`
through the now-stale cursor.

Upstream fix `9e0b4b1ab5ef`, live in our pinned 17.0. Citations and quoted lines
in `PROVENANCE.md`.

## What the run establishes

```
  the Change slab context was created                        ok
    chunk size is the real sizeof(ReorderBufferChange) = 80

  before the return, the cursor sees its own LSN             ok
  the returned chunk is handed straight back to the next record ok
    64 of 64 free/allocate rounds returned the identical address

  slab reuse is deterministic, not incidental                ok
    stale read of change->lsn gave 00000002DEADBEEF
    the record it was taken from held 0000000100000001;
    the successor now in that chunk holds 00000002DEADBEEF
```

Progress is reported at an LSN the record never had — it belongs to the record
allocated after it.

## Why this row is worth having twice over

**It is deterministic.** `slab.c` has one chunk size per context and a LIFO free
list (`:731-732` pushes, `:277-286` pops), with no size classes and no
coalescing. The driver checks the address over 64 rounds and gets the same
pointer every time. The AllocSet cases in this directory cannot promise that —
there the reuse depends on the size class and on what else the workload asked
for.

**It is the first case with a Capstone arm available.** Slab was ported to the
Sublet discipline in `postgresql-17.0-0005-slab-sublet-lifetimes.patch`. Before
that, a Slab-allocated defect could only ever be run unprotected.

## Fidelity, stated plainly

The allocator, the context parameters, the record type and both the allocation
and the free are real and unmodified. The **decoding loop is modelled** — the
speculative-insert protocol, the transaction and the output plugin — because
reaching it needs a walsender, a replication slot and WAL.

That makes this a weaker provenance tier than the bitmapset cases, where the
consumer source itself is compiled. `PROVENANCE.md` has the line-by-line split of
what is real and what is not. To confirm the defect in a real backend rather than
in this driver, upstream's own recipe is to hardcode `CHANGES_THRESHOLD` to 1 and
run `test_decoding`'s `ddl` test under Valgrind.

## Tool coverage

No claim is made from ASan's silence: `pfree` never calls `free`, so ASan has no
event and cannot fire either way.

One caveat specific to this case — the driver allocates the successor *before*
the stale read, which makes the chunk addressable to Valgrind again. To get a
Valgrind verdict the free and the read have to be adjacent, without the
intervening allocation. That arm has not been run: no valgrind on this host, and
the tree is not built with `USE_VALGRIND`.

## Not yet done

- The **Valgrind detect arm**, in the ordering described above.
- The **Capstone arm**, now that slab is ported. Needs a domain build; no
  Capstone toolchain on the machine this was written on.

---

**Tier: MODEL consumer, real allocator.** Weaker than the bitmapset cases, and
deliberately labelled so. There the consumer source (`bitmapset.c`) is compiled
unmodified; here it cannot be, so the decoding loop is reconstructed from the
upstream diff while everything below it stays real.

| | |
|---|---|
| Allocator | **real** — `slab.c`, `mcxt.c`, `aset.c` compiled unmodified from the pinned 17.0 release |
| Context parameters | **real** — `SlabContextCreate(parent, "Change", SLAB_DEFAULT_BLOCK_SIZE, sizeof(ReorderBufferChange))`, as `reorderbuffer.c:329` creates it |
| Record type | **real** — `ReorderBufferChange` from `replication/reorderbuffer.h`, so the chunk size is the real 80 bytes and `lsn` is the real first field |
| Allocation call | **real** — `MemoryContextAlloc(change_context, sizeof(ReorderBufferChange))`, as `reorderbuffer.c:485` |
| Free call | **real** — `pfree`, as `ReorderBufferReturnChange` reaches it at `reorderbuffer.c:557` |
| Decoding loop | **modelled** — the speculative-insert protocol, the transaction, the output plugin and the 100-change threshold |

Reaching the real loop needs a walsender, a replication slot and WAL. None of
that changes what the manager does with the chunk, which is what the case is
about; but the reduction is real and is why this row is tiered below the
bitmapset ones.

## Upstream defect

- **Fix:** `9e0b4b1ab5ef` — *"Fix use-after-free with INSERT ON CONFLICT changes in reorderbuffer.c"*, 2025-08-02, back-patched to 16. Four commits across branches, one defect.
- **File:** `src/backend/replication/logical/reorderbuffer.c`
- **Introduced by:** `8c58624df462`, which added the periodic progress update.
- **CVE:** `NO VERIFIED CVE`. No bug number in the trailer either.
- **Live in our pin:** yes — verified in `REL_17_0`; nothing is reverted.

## The defect, by line in the pinned release

The `INTERNAL_SPEC_CONFIRM` arm aliases the loop cursor onto the delayed record
(`:2208-2219`):

```c
change = specinsert;
```

the common exit returns that record to its context (`:2318-2322`, reaching
`pfree` at `:557`):

```c
change_done:
    ReorderBufferReturnChange(rb, specinsert, true);
```

and every hundredth change the loop reports progress through the cursor
(`:2496`):

```c
rb->update_progress_txn(rb, txn, change->lsn);
```

The fix passes the previously recorded LSN instead — one line.

## Why Slab makes this the corpus's deterministic row

`slab.c` keeps one chunk size per context and a LIFO free list threaded through
the block: `SlabFree` pushes at `:731-732`, `SlabAlloc` pops at `:277-286`. There
are no size classes, no coalescing and no block merging, so a free followed by an
allocation of that context returns the identical address unconditionally.

The driver checks this over 64 rounds rather than once, and all 64 return the
same pointer. With AllocSet the same experiment depends on the size class and on
what else the workload allocated; with Slab it is a property of the allocator.

## Reproducing it upstream

Masahiko Sawada's recipe, quoted in the commit discussion: hardcode
`CHANGES_THRESHOLD` to 1 and run `test_decoding`'s `ddl` test under Valgrind.
That is the route to confirming the defect in a real backend rather than in this
driver, and it is not what this case does.

## Tool coverage

No claim is made from AddressSanitizer. `pfree` does not call `free`, so ASan has
no event and cannot fire; its silence measures nothing. Valgrind can fire,
because PostgreSQL hand-wrote the mempool annotations — `VALGRIND_MEMPOOL_FREE`
in `pfree` itself (`mcxt.c:1531`), `VALGRIND_MEMPOOL_ALLOC` on every `palloc`
(`mcxt.c:1201`), `VALGRIND_CREATE_MEMPOOL` per context (`mcxt.c:1137`) — all
behind `USE_VALGRIND`.

Note for this case specifically: the driver allocates the successor before the
stale read, which makes the chunk addressable to Valgrind again. A Valgrind run
therefore needs the free-then-read ordering without the intervening allocation to
report anything. The runner's Valgrind arm has not been run here — this machine
has no valgrind and the tree is not built with `USE_VALGRIND`.

## Capstone

Slab is ported to the Sublet discipline as of
`patches/postgresql-17.0-0005-slab-sublet-lifetimes.patch`, so this case has a
protected arm available for the first time. It has not been run: that needs a
domain build and there is no Capstone toolchain on this machine.
