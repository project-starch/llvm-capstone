/* The entry point, the root context, the error jump target and the labelled
 * probe. One program per case, as the contract requires: a capability fault
 * ends the run, so a case that provokes one cannot report results beside it.
 */
#include "corpus.h"
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

extern sigjmp_buf ro_error_jump;
extern volatile sig_atomic_t ro_error_armed;

MemoryContext ro_root;

/* Release accounting, keyed by payload. Small and linear on purpose: a case
 * owns one or two resources and the table is evidence, not a data structure. */
#define RO_SLOTS 8
static const void *ro_seen[RO_SLOTS];
static unsigned ro_count[RO_SLOTS];

void
ro_note_release(const void *payload)
{
	for (unsigned i = 0; i < RO_SLOTS; i++)
	{
		if (ro_seen[i] == payload)
		{
			ro_count[i]++;
			return;
		}
		if (ro_seen[i] == NULL)
		{
			ro_seen[i] = payload;
			ro_count[i] = 1;
			return;
		}
	}
	ro_give_up(90);
}

unsigned
ro_release_count(const void *payload)
{
	for (unsigned i = 0; i < RO_SLOTS; i++)
		if (ro_seen[i] == payload)
			return ro_count[i];
	return 0;
}

/* The labelled free. A double-release defect has no read probe -- the second
 * free IS the invalid operation -- so this is the frame a fault must land in
 * for a run to attribute it to the defect. */
void
ro_free_probe(void *payload)
{
	free(payload);
}

void
ro_mark(void)
{
	fprintf(stderr, "RO_DEFECT case=%u ready\n", ro_case_number);
	fflush(stderr);
}

_Noreturn void
ro_raise(const char *what)
{
	fprintf(stderr, "  [raise] %s\n", what);
	fflush(stderr);
	if (ro_error_armed)
		siglongjmp(ro_error_jump, 1);
	fprintf(stderr, "  [raise] no handler armed\n");
	exit(70);
}

_Noreturn void
ro_give_up(unsigned long code)
{
	fprintf(stderr, "RO_DEFECT case=%u gave up code=%lu\n", ro_case_number,
			code);
	exit((int) (code & 0x7f));
}

/* How a backend reaches the second release.
 *
 * CommitTransaction (or any caller) releases the owner. A release callback
 * raises an error. The error unwinds to the abort handler, and
 * AbortTransaction releases the SAME owner again -- resowner.c's own comment
 * for the fixed code says so: "the subsequent abort cleanup will come back
 * here and try to release that item again".
 */
unsigned
ro_release_with_abort_cleanup(ResourceOwner owner)
{
	unsigned	errors = 0;

	ro_error_armed = 1;
	if (sigsetjmp(ro_error_jump, 1) == 0)
	{
		fprintf(stderr, "  [pass 1] ResourceOwnerRelease, isCommit=true\n");
		ResourceOwnerRelease(owner, RESOURCE_RELEASE_BEFORE_LOCKS, true, true);
	}
	else
	{
		errors++;
		fprintf(stderr, "  [abort cleanup] error escaped pass 1; releasing the "
				"same owner again\n");
		/* AbortTransaction's own call: isCommit=false. If this raises too, the
		 * backend would loop; we take the error once and stop, which is enough
		 * to show the second release happened. */
		if (sigsetjmp(ro_error_jump, 1) == 0)
			ResourceOwnerRelease(owner, RESOURCE_RELEASE_BEFORE_LOCKS, false,
								 true);
		else
		{
			errors++;
			fprintf(stderr, "  [abort cleanup] and the second release raised "
					"again -- the error cleanup loop upstream warns about\n");
		}
	}
	ro_error_armed = 0;
	return errors;
}

int
main(void)
{
	/* The same root the mmgr corpus builds, and for the same reason:
	 * MemoryContextInit() wants more of a backend than this program links, so
	 * the root is created directly and installed as both globals. resowner.c
	 * allocates its owners out of TopMemoryContext. */
	ro_root = AllocSetContextCreateInternal(NULL, "ro-root", 0, 2048, 8192);
	TopMemoryContext = CurrentMemoryContext = ro_root;

	fprintf(stderr, "RO_DEFECT case=%u start\n", ro_case_number);
	ro_case_run();
	fprintf(stderr, "RO_DEFECT case=%u completed\n", ro_case_number);
	return 0;
}
