/* reorderbuffer: an LSN read out of a change record that was already returned
 * to its Slab context.
 *
 * Upstream commit 9e0b4b1ab5ef, live in the pinned 17.0 release. In
 * ReorderBufferProcessTXN, an INSERT ... ON CONFLICT arrives as a delayed
 * INTERNAL_SPEC_INSERT and is only processed when its INTERNAL_SPEC_CONFIRM
 * turns up. That arm aliases the loop's cursor onto the delayed record:
 *
 *     change = specinsert;                       // reorderbuffer.c:2208-2219
 *
 * and the common exit then hands the delayed record back:
 *
 *   change_done:
 *     ReorderBufferReturnChange(rb, specinsert, true);   // :2318-2322 -> :557 pfree
 *
 * leaving `change` pointing at a returned chunk. Every hundredth change the
 * loop reports progress through it:
 *
 *     rb->update_progress_txn(rb, txn, change->lsn);     // :2496
 *
 * so the LSN comes out of memory the manager has already taken back. The fix
 * passes the previous LSN instead; it is one line.
 *
 * WHY THIS ONE IS THE DETERMINISTIC MEMBER OF THE CORPUS
 *
 * Change records come from a Slab context, not an AllocSet:
 *
 *     buffer->change_context = SlabContextCreate(new_ctx, "Change",
 *                                 SLAB_DEFAULT_BLOCK_SIZE,
 *                                 sizeof(ReorderBufferChange));   // :329
 *     ... MemoryContextAlloc(rb->change_context, sizeof(ReorderBufferChange));  // :485
 *
 * Slab has one chunk size per context, a LIFO free list threaded through the
 * block (slab.c:731-732 pushes, slab.c:277-286 pops) and no size classes or
 * coalescing. So a free followed by an allocation returns the IDENTICAL address
 * every time, rather than depending on what else the workload asked for. This
 * driver asserts that over many rounds rather than once, because "the same
 * pointer came back" is only interesting if it is a property of the allocator
 * and not an accident of one run.
 *
 * FIDELITY, STATED PLAINLY
 *
 * The allocator is real: slab.c and mcxt.c compiled unmodified from the pinned
 * release, created with the real parameters, and the record is the real
 * ReorderBufferChange from replication/reorderbuffer.h, so the chunk size is
 * the real one (80 bytes). What is modelled is the decoding loop -- the
 * speculative-insert protocol, the transaction and the output plugin -- because
 * reaching it needs a walsender, a replication slot and WAL. None of that
 * changes what the manager does with the chunk, which is the subject.
 *
 * This is therefore a weaker provenance tier than the bitmapset cases, where
 * the consumer source itself is compiled. PROVENANCE.md says so.
 *
 * On ASan: it cannot fire here, for the same reason as every case in this
 * directory -- pfree does not call free. The runner says so and claims nothing
 * from its silence.
 */
#include "postgres.h"

#include "replication/reorderbuffer.h"
#include "utils/memutils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Distinguishable LSNs: the delayed record's, and its successor's. */
#define LSN_SPEC      0x0000000100000001UL
#define LSN_SUCCESSOR 0x00000002DEADBEEFUL

/* How many free/allocate rounds to check the address is not an accident. */
#define ROUNDS 64

static int failures = 0;

static void
check(int ok, const char *what)
{
	printf("%-58s %s\n", what, ok ? "ok" : "FAILED");
	if (!ok)
		failures++;
}

static void
control_plain_malloc_uaf(void)
{
	volatile unsigned char *p = malloc(64);

	if (p == NULL)
		exit(2);
	p[0] = 0x41;
	free((void *) p);
	printf("control: read 0x%02x from freed malloc memory\n", p[0]);
}

int
main(int argc, char **argv)
{
	MemoryContext change_context;
	ReorderBufferChange *specinsert;
	ReorderBufferChange *change;	/* the loop cursor, aliased onto specinsert */
	ReorderBufferChange *successor;
	const void *freed_at;
	int			same = 0;
	int			i;

	if (argc > 1 && strcmp(argv[1], "--control") == 0)
	{
		control_plain_malloc_uaf();
		return 0;
	}

	/* The manager standing on its own; then the real context, real parameters. */
	TopMemoryContext = CurrentMemoryContext =
		AllocSetContextCreateInternal(NULL, "corpus",
									  ALLOCSET_DEFAULT_MINSIZE,
									  ALLOCSET_DEFAULT_INITSIZE,
									  ALLOCSET_DEFAULT_MAXSIZE);

	change_context = SlabContextCreate(TopMemoryContext, "Change",
									   SLAB_DEFAULT_BLOCK_SIZE,
									   sizeof(ReorderBufferChange));
	check(change_context != NULL, "the Change slab context was created");
	printf("  chunk size is the real sizeof(ReorderBufferChange) = %zu\n\n",
		   sizeof(ReorderBufferChange));

	/* The delayed speculative insert. */
	specinsert = MemoryContextAlloc(change_context, sizeof(*specinsert));
	memset(specinsert, 0, sizeof(*specinsert));
	specinsert->lsn = LSN_SPEC;
	freed_at = (const void *) specinsert;

	/* INTERNAL_SPEC_CONFIRM: the cursor is aliased onto the delayed record. */
	change = specinsert;
	check(change->lsn == LSN_SPEC, "before the return, the cursor sees its own LSN");

	/* change_done: ReorderBufferReturnChange(rb, specinsert, true) -> pfree */
	pfree(specinsert);

	/* The next change record the loop takes. Slab hands back the same chunk. */
	successor = MemoryContextAlloc(change_context, sizeof(*successor));
	memset(successor, 0, sizeof(*successor));
	successor->lsn = LSN_SUCCESSOR;

	check((const void *) successor == freed_at,
		  "the returned chunk is handed straight back to the next record");

	/*
	 * Slab's reuse is a property of the allocator, not of this run: one chunk
	 * size, LIFO free list, no size classes. Check it repeats.
	 */
	for (i = 0; i < ROUNDS; i++)
	{
		ReorderBufferChange *r;

		pfree(successor);
		r = MemoryContextAlloc(change_context, sizeof(*r));
		if ((const void *) r == freed_at)
			same++;
		successor = r;
	}
	memset(successor, 0, sizeof(*successor));
	successor->lsn = LSN_SUCCESSOR;
	printf("  %d of %d free/allocate rounds returned the identical address\n\n",
		   same, ROUNDS);
	check(same == ROUNDS, "slab reuse is deterministic, not incidental");

	/* :2496 -- progress reported through the stale cursor. */
	printf("  stale read of change->lsn gave %016lX\n", (unsigned long) change->lsn);
	printf("  the record it was taken from held %016lX;\n"
		   "  the successor now in that chunk holds %016lX\n\n",
		   (unsigned long) LSN_SPEC, (unsigned long) LSN_SUCCESSOR);

	check(change->lsn == LSN_SUCCESSOR,
		  "the stale read returns the SUCCESSOR's LSN, not its own");
	check(change->lsn != LSN_SPEC,
		  "progress is reported at an LSN the record never had");

	printf("%s\n", failures == 0
		   ? "VERDICT: stale-read-returns-successor-lsn"
		   : "VERDICT: unreproduced");
	return failures == 0 ? 0 : 1;
}
