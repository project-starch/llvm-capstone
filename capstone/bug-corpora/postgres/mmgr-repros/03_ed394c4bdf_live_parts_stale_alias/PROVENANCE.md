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

---

**Tier: LITERAL-traceable.** The defective allocator is the real one, unmodified;
the defective data structure (`bitmapset.c`) is the real one, unmodified; the
lifetime is ended by the real upstream call. Only the *caller* — the planner
loop in `expand_partitioned_rtentry()` — is reduced to its essence, because it
needs a live backend, a partitioned table and a concurrently dropped partition.

## Upstream defect

- **Fix:** `ed394c4bdf12` — *"Fix possible use after free in expand_partitioned_rtentry()"*, 2025-08-30, back-patched through 15. Five commits across branches, one defect.
- **File:** `src/backend/optimizer/util/inherit.c`
- **Reported by:** an external contributor via pgsql-bugs; no bug number in the trailer.
- **CVE:** `NO VERIFIED CVE`. PostgreSQL assigns CVEs sparingly and did not here.
- **Live in our pin:** yes. Verified in `REL_17_0:src/backend/optimizer/util/inherit.c` — the pre-fix shape is present, so nothing is reverted to reproduce it.

## The vulnerable code, quoted from the pinned release

`inherit.c:359` takes two aliases to one Bitmapset:

```c
relinfo->live_parts = live_parts = prune_append_rel_partitions(relinfo);
```

`:381` iterates the **local**:

```c
i = -1;
while ((i = bms_next_member(live_parts, i)) >= 0)
```

and `:397`, the concurrently-dropped-partition path, assigns through the
**field** only:

```c
childrel = try_table_open(childOID, lockmode);
if (childrel == NULL)
{
    relinfo->live_parts = bms_del_member(relinfo->live_parts, i);
    continue;
}
```

## The fix

```diff
-	Bitmapset  *live_parts;
...
-	relinfo->live_parts = live_parts = prune_append_rel_partitions(relinfo);
+	relinfo->live_parts = prune_append_rel_partitions(relinfo);
...
-	while ((i = bms_next_member(live_parts, i)) >= 0)
+	while ((i = bms_next_member(relinfo->live_parts, i)) >= 0)
```

Three insertions, four deletions: the local alias is removed and every read goes
through the field, so there is only ever one alias.

## Why the free is invisible

`bms_del_member` releases the set when its last member goes
(`src/backend/nodes/bitmapset.c`):

```c
		/* the set is now empty */
		pfree(a);
		return NULL;
```

and `pfree` on an AllocSet chunk does not reach `malloc`. It pushes the chunk
onto the context's size-class free list (`aset.c:1139-1143`), and the next
`palloc` of that class pops the same address (`aset.c:1000-1013`). The header of
`aset.c` states it outright at `:20-22`:

> "AllocSetFree() normally doesn't free() memory really. It just add's the
> free'd area to some list for later reuse by AllocSetAlloc()."

## What the driver changes, and what it does not

**Not changed:** `aset.c`, `mcxt.c` and the rest of the manager; `bitmapset.c`;
the allocation and free calls; the size class; the order of operations.

**Reduced:** `prune_append_rel_partitions()` becomes `bms_add_member`, and
`try_table_open()` returning NULL becomes taking the drop path directly. Neither
touches the allocator.

**Added, to make the result observable rather than merely true:** a second
`bms_add_member` between the free and the stale read, standing for the unrelated
code that claims the recycled chunk. Without it the stale read would return
stale-but-intact bytes and the run would prove only that nothing crashed. With
it, the read returns another live object's member.

## Upstream's own mitigation, and why it is not the oracle

`bitmapset.c:23-27` documents `REALLOCATE_BITMAPSETS`, a build option that
reallocates and frees the set on every modification specifically

> "To help find bugs caused by callers failing to record the return value of the
> function which manipulates an existing set"

It is off by default and debug-only. It would make this defect deterministic; it
would **not** make it visible to a malloc-level tool, because the reallocation is
still `palloc`/`pfree`.

`REALLOCATE_BITMAPSETS` is the second of three hand-written, opt-in, debug-only
mechanisms this one defect runs past. The others are `MEMORY_CONTEXT_CHECKING`,
whose double-`pfree` detector arrived only in March 2026 (`0c8b4e9cfc`) and was
kept out of production builds because *"that adds measurable overhead"*, and the
Valgrind mempool annotations in `mcxt.c` and `aset.c`, compiled out unless
`USE_VALGRIND` is defined.

That last one is why this case makes **no claim from AddressSanitizer's
silence**. ASan cannot fire here by construction — no `malloc` or `free` occurs
between the free and the read — so its silence measures nothing. Valgrind *can*
fire, precisely because somebody wrote those annotations by hand. The
discriminating question is not whether a tool can see into a nested allocator but
who taught it to, for which tool, and in which build. See the README.
