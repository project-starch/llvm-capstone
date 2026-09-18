# Provenance

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
still `palloc`/`pfree`. That is why this case's control is a plain `malloc`
use-after-free instead: it proves the harness can see the class it *can* see.
