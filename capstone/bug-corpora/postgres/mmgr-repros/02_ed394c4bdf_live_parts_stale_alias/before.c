/* expand_partitioned_rtentry(): a Bitmapset freed through one alias and read
 * through another, on PostgreSQL's own AllocSet.
 *
 * The defect is upstream commit ed394c4bdf12, live in the pinned 17.0 release.
 * inherit.c takes two aliases to one set and then frees through only one of
 * them:
 *
 *     relinfo->live_parts = live_parts = prune_append_rel_partitions(relinfo);
 *     i = -1;
 *     while ((i = bms_next_member(live_parts, i)) >= 0)      // alias A
 *     {
 *         childrel = try_table_open(childOID, lockmode);
 *         if (childrel == NULL)
 *         {
 *             relinfo->live_parts =                          // alias B
 *                 bms_del_member(relinfo->live_parts, i);
 *             continue;
 *         }
 *         ...
 *     }
 *
 * bitmapset.c frees the set when its last member goes:
 *
 *     /-* the set is now empty *-/
 *     pfree(a);
 *     return NULL;
 *
 * So alias B becomes NULL, which is safe, and alias A keeps pointing at a
 * chunk that pfree has put on the AllocSet's size-class free list. The loop
 * then reads it.
 *
 * WHAT THIS DRIVER SHOWS, AND WHAT IT DELIBERATELY DOES NOT
 *
 * The whole point of the case is that nothing goes wrong in the way a
 * memory-safety tool recognises. pfree does not call free(): the chunk goes on
 * set->freelist[fidx] (aset.c:1139-1143) and the next palloc of that size class
 * gets the same address back (aset.c:1000-1013). The block it lives in is still
 * malloc'd and still readable. So the stale read succeeds, in bounds, on a live
 * object -- someone else's.
 *
 * To make that visible rather than merely true, the driver claims the freed
 * chunk with a second Bitmapset before reading through the stale alias. The
 * claimant is a well-formed set with a different member, so the stale read
 * returns a wrong answer instead of crashing: the loop is told the partition
 * it already deleted is still there, under a different index.
 *
 * ON ASAN, AND WHY THIS CASE CLAIMS NOTHING FROM IT
 *
 * ASan instruments malloc and free, and neither happens here between the free
 * and the read. It therefore cannot fire, and its silence says nothing about
 * PostgreSQL -- only that AllocSet is an allocator above malloc, which the code
 * already told us. The malloc use-after-free below is a control on the BINARY,
 * proving it really is instrumented; it is not a control on the subject, whose
 * memory it never touches, and it does not make the subject's silence evidence.
 *
 * The tool that can fire on the subject is Valgrind, because PostgreSQL wrote
 * the mempool annotations for it by hand (mcxt.c:422, mcxt.c:1201,
 * aset.c:879-881), all behind USE_VALGRIND. The runner has that arm.
 */
#include "postgres.h"

#include "nodes/bitmapset.h"
#include "utils/memutils.h"
#include "utils/memutils_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The member the partition loop starts with, and the one its replacement
 * carries. Any two distinct values do; these are just easy to read. */
#define OURS   3
#define THEIRS 7

static int failures = 0;

static void
check(int ok, const char *what)
{
	printf("%-58s %s\n", what, ok ? "ok" : "FAILED");
	if (!ok)
		failures++;
}

/*
 * The control. ASan must report this, or its silence about the subject below
 * is a statement about the instrument and not about PostgreSQL.
 *
 * Only run when the harness asks for it, because under ASan it is fatal by
 * design and would stop the subject from running at all.
 */
static void
control_plain_malloc_uaf(void)
{
	volatile unsigned char *p = malloc(64);

	if (p == NULL)
		exit(2);
	p[0] = 0x41;
	free((void *) p);
	/* The access a malloc-level tool exists to catch. */
	printf("control: read 0x%02x from freed malloc memory (a tool that can see "
		   "this class must have reported by now)\n", p[0]);
}

int
main(int argc, char **argv)
{
	MemoryContext ctx;
	Bitmapset  *live_parts;		/* the local in expand_partitioned_rtentry */
	Bitmapset  *field;			/* relinfo->live_parts */
	Bitmapset  *claimant;
	const void *freed_at;
	const void *claimed_at;
	int			member;

	if (argc > 1 && strcmp(argv[1], "--control") == 0)
	{
		control_plain_malloc_uaf();
		return 0;
	}

	/* The manager, standing on its own: the first context is its own root. */
	ctx = AllocSetContextCreateInternal(NULL, "corpus",
										ALLOCSET_DEFAULT_MINSIZE,
										ALLOCSET_DEFAULT_INITSIZE,
										ALLOCSET_DEFAULT_MAXSIZE);
	TopMemoryContext = CurrentMemoryContext = ctx;

	/* relinfo->live_parts = live_parts = prune_append_rel_partitions(relinfo) */
	field = live_parts = bms_add_member(NULL, OURS);
	freed_at = (const void *) live_parts;
	check(live_parts != NULL, "the partition set was allocated");
	check(bms_next_member(live_parts, -1) == OURS,
		  "before the drop, the loop sees our own partition");

	/*
	 * try_table_open() returned NULL -- the partition was concurrently
	 * detached and dropped -- so the drop path runs. It assigns through the
	 * field only, exactly as 17.0 does.
	 */
	field = bms_del_member(field, OURS);
	check(field == NULL, "the field is emptied and nulled, which is safe");

	/*
	 * Nothing has been returned to malloc. The chunk is on the context's
	 * size-class free list, so the next same-size allocation gets it back.
	 * Here that is another partition set belonging to unrelated code.
	 */
	claimant = bms_add_member(NULL, THEIRS);
	claimed_at = (const void *) claimant;
	check(claimed_at == freed_at,
		  "the freed chunk is handed straight back to the next caller");

	/*
	 * The next turn of the loop, reading through the stale alias. In bounds,
	 * correctly typed, on a live object -- and the wrong one.
	 */
	member = bms_next_member(live_parts, -1);
	printf("\n  stale read through the dropped alias returned member %d\n", member);
	printf("  the loop had deleted %d and should see nothing; it sees %d,\n"
		   "  which belongs to the set allocated after the free\n\n", OURS, member);

	check(member == THEIRS,
		  "the stale read returns the NEW owner's data, not ours");
	check(member != -1,
		  "the loop is told a deleted partition is still live");

	printf("%s\n", failures == 0
		   ? "VERDICT: stale-read-returns-other-object"
		   : "VERDICT: unreproduced");
	return failures == 0 ? 0 : 1;
}
