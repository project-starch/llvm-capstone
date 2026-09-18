# `ed394c4bdf` — a partition set freed through one alias, read through another

A use-after-free in PostgreSQL's planner that no malloc-level tool can see,
because the memory never goes back to `malloc`.

    bash ../run-host-repros.sh ed394c4bdf_live_parts_stale_alias

## The defect

`expand_partitioned_rtentry()` keeps two aliases to one `Bitmapset` — a local and
a field on the `RelOptInfo`. When a partition turns out to have been dropped
concurrently it removes that partition through the **field**, and
`bms_del_member` frees the set once the last member goes. The field becomes
NULL, which is safe. The **local** still points at the freed chunk, and the next
turn of the loop reads it.

Upstream fix `ed394c4bdf12`, live in our pinned 17.0. Full citation and the
quoted hunks are in `PROVENANCE.md`.

## What the run shows

```
control: ASan reported heap-use-after-free -- the harness can see this class

  the partition set was allocated                            ok
  before the drop, the loop sees our own partition           ok
  the field is emptied and nulled, which is safe             ok
  the freed chunk is handed straight back to the next caller ok

    stale read through the dropped alias returned member 7
    the loop had deleted 3 and should see nothing; it sees 7,
    which belongs to the set allocated after the free

  plain          rc=0 verdict=stale-read-returns-other-object
  under ASan     rc=0 verdict=stale-read-returns-other-object  asan-silent=yes
  RESULT: REPRODUCED, and invisible to ASan
```

Three things happen, and the third is the point:

1. `pfree` puts the chunk on the context's size-class free list. It does not call
   `free`.
2. The next `palloc` of that size hands **the same address** back to unrelated
   code — the driver checks the two pointers are equal, so this is measured, not
   asserted.
3. The stale read then succeeds. It is in bounds, correctly typed, on a live
   object, and returns that object's data. The loop is told a partition it just
   deleted is still there, under someone else's index.

There is no moment at which a tool watching `malloc` and `free` could have
intervened, because no such call happens between the free and the read.

## Why the control matters

The finding is a silence, and a silence is worth nothing until the instrument is
shown to make noise. So the same binary, under the same ASan, first commits a
plain `malloc`/`free`/use. ASan reports it. Only then is ASan's silence about the
subject evidence about PostgreSQL rather than about ASan.

`run-host-repros.sh` exits **75 with no verdict** if the control fails to fire.

## What is real here

The allocator is PostgreSQL's own, compiled unmodified from the pinned release —
`aset.c`, `mcxt.c`, `generation.c`, `slab.c`, `bump.c`, `alignedalloc.c`,
`memdebug.c`. So is `bitmapset.c`, where the `pfree` lives. The planner loop
around them is reduced to its essence, because reproducing it in place needs a
running backend and a partition dropped concurrently by another session, and
neither changes what the allocator does. `PROVENANCE.md` lists exactly what was
reduced.

## Not yet done

The Capstone arm. Under the Sublet discipline this allocation is a sub-pool
carve, the `pfree` is a revocation, and the stale read should take a capability
fault instead of returning someone else's partition. That needs a domain build,
which needs a Capstone toolchain; there is none on the machine this was written
on. The native half stands on its own as the "before".
