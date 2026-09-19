# `ed394c4bdf` — a partition set freed through one alias, read through another

A use-after-free in PostgreSQL's planner, reproduced against PostgreSQL's own
unmodified memory manager.

    bash ../run-host-repros.sh ed394c4bdf_live_parts_stale_alias

## The defect

`expand_partitioned_rtentry()` keeps two aliases to one `Bitmapset` — a local and
a field on the `RelOptInfo`. When a partition turns out to have been dropped
concurrently it removes that partition through the **field**, and
`bms_del_member` frees the set once the last member goes. The field becomes
NULL, which is safe. The **local** still points at the freed chunk, and the next
turn of the loop reads it.

Upstream fix `ed394c4bdf12`, live in our pinned 17.0. Citations and quoted hunks
are in `PROVENANCE.md`.

## What the run establishes

```
  the freed chunk is handed straight back to the next caller ok

    stale read through the dropped alias returned member 7
    the loop had deleted 3 and should see nothing; it sees 7,
    which belongs to the set allocated after the free
```

Two things, both measured rather than asserted:

1. The address `pfree` released and the address the next same-size `palloc`
   returned are **equal**. The driver compares the two pointers. No `free()` is
   involved: the chunk goes on the context's size-class free list
   (`aset.c:1139-1143`) and comes back from `aset.c:1000-1013`.
2. The stale read then succeeds — in bounds, correctly typed, on a live object,
   and returns **that object's** data. The loop is told a partition it just
   deleted is still there, under someone else's index.

## What the run does NOT establish

**Nothing about tool coverage.** An earlier version of this case reported
"invisible to ASan" as though that were the finding. It is not, and the claim is
withdrawn.

ASan instruments `malloc` and `free`. The chunk here passes through neither, so
ASan has no event, cannot fire, and its silence restates how AllocSet works
rather than measuring anything. The malloc use-after-free control does not
rescue it: that control proves ASan sees malloc faults — which was never in
doubt — on memory this defect never touches. A control has to be able to fire
*on the subject's own terms* to make the subject's silence mean something, and
this one cannot.

So the ASan arm is kept only to show the verdict is unchanged under
instrumentation. It is labelled uninformative in the runner's output.

## The arm that would discriminate

Valgrind, because **PostgreSQL hand-taught it about the nested allocator**:

| | |
|---|---|
| `mcxt.c:422`, `:1137` | `VALGRIND_CREATE_MEMPOOL` per context |
| `mcxt.c:1201` and friends | `VALGRIND_MEMPOOL_ALLOC` on every `palloc` |
| `aset.c:879-881` | `VALGRIND_MAKE_MEM_NOACCESS` on a freed chunk's free-list link |

All of it compiles to `do {} while (0)` unless `USE_VALGRIND` is defined
(`memdebug.h:20-33`).

That is the real shape of the problem, and it is sharper than "tools can't see
nested allocators". A nested allocator is opaque to generic tooling **until
somebody writes the annotations by hand** — for one tool, in a debug build, per
allocator. PostgreSQL did that work for Valgrind. Nobody did it for ASan. And
`generation.c`, `slab.c` and `bump.c` each need their own.

The runner runs this arm when a `valgrind` binary and a `USE_VALGRIND` tree are
both present, and reports **SKIPPED** otherwise — a skipped arm is not a passing
arm. It has **not been run**: the machine this was written on has no valgrind.

## What is real here

PostgreSQL's own allocator, compiled unmodified from the pinned release —
`aset.c`, `mcxt.c`, `generation.c`, `slab.c`, `bump.c`, `alignedalloc.c`,
`memdebug.c` — and `bitmapset.c`, where the `pfree` lives. Only the planner loop
is reduced, because reproducing it in place needs a running backend and a
partition dropped by another session, and neither changes what the allocator
does. `PROVENANCE.md` lists exactly what was reduced.

## Not yet done

- The **Valgrind arm**, above. This is the one that turns the case from a
  reproduction into a statement about coverage.
- The **Capstone arm**. Under the Sublet discipline the `pfree` is a revocation
  and the stale read should take a capability fault instead of answering. Needs a
  domain build; there is no Capstone toolchain on the machine this was written
  on.
