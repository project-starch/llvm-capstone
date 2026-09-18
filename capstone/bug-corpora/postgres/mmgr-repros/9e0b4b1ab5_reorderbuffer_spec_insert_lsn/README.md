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
