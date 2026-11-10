#include "corpus.h"
#include <stdlib.h>
#include <stdio.h>

/* 011eedcdc3 -- "Fix unsafe order of operations in ResourceOwnerReleaseAll()".
 *
 * ResourceOwnerReleaseAll calls the kind's ReleaseResource for an item and
 * only then decrements the count.  If ReleaseResource raises an error, the
 * item is still in the owner's array, and abort cleanup releases the owner
 * again -- so the same item is released twice.
 *
 * resowner.c is the subject and is real.  What this case supplies is a
 * resource KIND, which is what every consumer of resowner.c supplies.  Its
 * ReleaseResource frees the payload and then fails, which is the order
 * pgcrypto's really has (case 01) and the order any release that does part of
 * its work before failing has.
 */

static unsigned pass;

static void
ReleaseThing(Datum res)
{
	void	   *payload = DatumGetPointer(res);

	ro_note_release(payload);
	pass++;
	fprintf(stderr, "  [ReleaseThing] pass %u on %p\n", pass, payload);

	/* The release does part of its work ... */
#ifdef RO_NEGATIVE_CONTROL
	/* ... and the control frees it only on the first pass. Every other
	 * operation, in both passes, is byte for byte what the defect does: the
	 * item is still released twice, the error is still raised, the abort
	 * cleanup still runs. Only the second free is withheld. */
	if (pass == 1)
		ro_free_probe(payload);
#else
	ro_free_probe(payload);		/* second pass: the double free */
#endif

	/* ... and then fails. In a backend this is any elog(ERROR) a release can
	 * reach; case 01 is the one upstream names, where the error comes out of
	 * resowner.c itself. */
	if (pass == 1)
		ro_raise("ReleaseThing failed after freeing its payload");
}

static const ResourceOwnerDesc thing_desc = {
	.name = "corpus thing",
	.release_phase = RESOURCE_RELEASE_BEFORE_LOCKS,
	.release_priority = RELEASE_PRIO_FIRST,
	.ReleaseResource = ReleaseThing,
	.DebugPrint = NULL,
};

RO_CASE(0) {
	ResourceOwner owner;
	void	   *payload;
	unsigned	errors;

	owner = ResourceOwnerCreate(NULL, "corpus owner");

	/* The payload is libc's, as pgcrypto's EVP_CIPHER_CTX is: the object that
	 * gets freed twice does not come from a memory context. */
	payload = malloc(64);
	CHECK(payload != NULL, 10);

	ResourceOwnerEnlarge(owner);
	ResourceOwnerRemember(owner, PointerGetDatum(payload), &thing_desc);

	ro_mark();
	errors = ro_release_with_abort_cleanup(owner);

	fprintf(stderr, "  releases of %p: %u (errors escaped: %u)\n",
			payload, ro_release_count(payload), errors);

	/* The claim, stated without a sanitizer: resowner released one item twice.
	 * At the pin there is nothing to stop it -- the count was never decremented
	 * before the callback ran. */
	CHECK(errors >= 1, 11);
	CHECK(ro_release_count(payload) == 2, 12);
}
