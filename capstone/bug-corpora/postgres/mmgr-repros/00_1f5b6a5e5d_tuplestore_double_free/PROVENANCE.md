# `1f5b6a5e5d` — a tuple freed by the write path and freed again by cleanup

A double free in PostgreSQL's tuplestore, reproduced against PostgreSQL's own
unmodified memory manager.

## The defect

`dumptuples()` writes each in-memory tuple out and frees it as it goes;
`WRITETUP` is `writetup_heap`, which ends with `FREEMEM` plus
`heap_free_minimal_tuple`. The loop clears `memtupdeleted` and `memtupcount`
only **after** it finishes. If a `BufFileWrite` inside `WRITETUP` throws part
way through, the counters still describe every slot as live while the first
few already point at freed chunks. `tuplestore_clear()` then walks
`memtupdeleted .. memtupcount` and frees them a second time.

## Upstream defect

Upstream fix `1f5b6a5e5d`, 2026-03-30, "Be more careful to preserve
consistency of a tuplestore", reported as bug #19438. The commit message
states the shape in its own words:

> Several places in tuplestore.c would leave the tuplestore data structure
> effectively corrupt if some subroutine were to throw an error. Notably, if
> WRITETUP() failed after some number of successful calls within
> dumptuples(), the tuplestore would contain some memtuples pointers that
> were apparently live entries but in fact pointed to pfree'd chunks.

- **CVE:** `NO VERIFIED CVE`. PostgreSQL assigns CVEs sparingly and did not here.
- **Live in our pin:** yes. The fix postdates the 17.0 release of 2024-09-26 by
  eighteen months, and the pre-fix shape is present in `REL_17_0`; nothing is
  reverted to reproduce it. Quoted below by line.

## The vulnerable code, quoted from the pinned release

`src/backend/utils/sort/tuplestore.c:1206`, the loop that frees:

```c
static void
dumptuples(Tuplestorestate *state)
{
	int			i;

	for (i = state->memtupdeleted;; i++)
	{
		...
		if (i >= state->memtupcount)
			break;
		WRITETUP(state, state->memtuples[i]);      /* :1222 -- frees it */
	}
	state->memtupdeleted = 0;                       /* :1225 -- only here */
	state->memtupcount = 0;                         /* :1226 */
}
```

`:1500`, what `WRITETUP` resolves to, and where the throw can come from:

```c
writetup_heap(Tuplestorestate *state, void *tup)
{
	...
	BufFileWrite(state->myfile, &tuplen, sizeof(tuplen));   /* :1511 */
	BufFileWrite(state->myfile, tupbody, tupbodylen);       /* :1512 */
	if (state->backward)
		BufFileWrite(state->myfile, &tuplen, sizeof(tuplen));

	FREEMEM(state, GetMemoryChunkSpace(tuple));             /* :1516 */
	heap_free_minimal_tuple(tuple);                         /* :1517 */
}
```

`:418`, the cleanup walk that frees a second time:

```c
tuplestore_clear(Tuplestorestate *state)
{
	...
	if (state->memtuples)
	{
		for (i = state->memtupdeleted; i < state->memtupcount; i++)
		{
			FREEMEM(state, GetMemoryChunkSpace(state->memtuples[i]));  /* :430 */
			pfree(state->memtuples[i]);                                /* :431 */
		}
	}
```

## The fix

The upstream hunk makes the bookkeeping follow the free immediately, and its
own comment says why:

```diff
 		WRITETUP(state, state->memtuples[i]);
+
+		/*
+		 * Increase memtupdeleted to track the fact that we just deleted that
+		 * tuple.  Think not to remove this on the grounds that we'll reset
+		 * memtupdeleted to zero below.  We might not reach that if some later
+		 * WRITETUP fails
```

## What is real here, and what is reduced

**Real:** the allocator. `aset.c`, `mcxt.c`, `slab.c`, `generation.c`,
`bump.c`, `alignedalloc.c` and `memdebug.c` from the pinned 17.0 release,
compiled unmodified but for the capability-ABI and Sublet patches the port
applies. The double free lands in the real manager, and the fault is the real
manager's.

**Reduced:** the consumer. Reaching `dumptuples()` in place needs a tuplestore
under memory pressure and a `BufFileWrite` that throws — a full backend and an
I/O failure. The case performs the allocator calls that sequence makes, in the
same order: two chunks of the same size class, one freed, then freed again.

## What the run establishes, and what it does not

This case has **no read probe**. The stale access IS the second `pfree`, so
there is no labelled load to require a fault at, and its oracle accepts a
fault anywhere — the only case in this corpus that does. Measured on
2026-09-20, the protected arm faults inside `GetMemoryChunkMethodID`, where
the manager reads the revoked chunk's header to dispatch the free, before any
of its own bookkeeping runs. The archived Capstone run of 2026-09-18 records
the same function for its `sublet` arm.

It does **not** establish that a double free is caught in general. It
establishes that this one is, on this allocator, at this pin.

## Not yet done

- No `before.c`, so this case has no host `native-detect` arm. ASan would be
  silent here for the reason the corpus runner documents — the chunk never
  passes through `malloc` or `free` — and Valgrind is the arm that would
  discriminate.
- The Capstone `spatial` and `sublet` arms have not been re-run since the
  corpus was split into one program per case.
