#include "corpus.h"
#include <stdlib.h>
#include <stdio.h>

/* 2aa6be6e64 -- "pgcrypto: avoid recursive ResourceOwnerForget()".
 *
 * The complete chain, with resowner.c real and raising the error itself.
 *
 * resowner.h's contract for ReleaseResource says: "The resource is implicitly
 * removed from the owner, the callback function doesn't need to call
 * ResourceOwnerForget."  pgcrypto's callback breaks it: at the pin
 * ResOwnerReleaseOSSLCipher hands the object straight to free_openssl_cipher
 * without clearing od->owner, so free_openssl_cipher's own
 * `if (od->owner != NULL) ResourceOwnerForgetOSSLCipher(...)` fires during
 * release.  Real resowner.c then raises
 *
 *     ResourceOwnerForget called for %s after release started
 *
 * and because ReleaseResource had already run (011eedcdc3, case 00) the item
 * is still in the array, so abort cleanup releases it a second time and
 * EVP_CIPHER_CTX_free runs twice on one handle.
 *
 * Transcribed from 17.5 contrib/pgcrypto/openssl.c:299-305 and :799-803.  The
 * EVP context is modelled by a libc allocation, which is what OpenSSL's
 * allocator is here; note the order, which is upstream's: the handle is freed
 * BEFORE the Forget that raises, so pfree(od) is never reached on either pass
 * and the object freed twice is the libc one.
 */

typedef struct OSSLCipher
{
	void	   *evp_ctx;		/* EVP_CIPHER_CTX_new() -- OpenSSL's allocator */
	ResourceOwner owner;
} OSSLCipher;

static void ResOwnerReleaseOSSLCipher(Datum res);

static const ResourceOwnerDesc osslcipher_resowner_desc = {
	.name = "pgcrypto OpenSSL cipher handle",
	.release_phase = RESOURCE_RELEASE_BEFORE_LOCKS,
	.release_priority = RELEASE_PRIO_FIRST,
	.ReleaseResource = ResOwnerReleaseOSSLCipher,
	.DebugPrint = NULL,
};

static void
free_openssl_cipher(OSSLCipher *od)
{
	/* EVP_CIPHER_CTX_free(od->evp_ctx) */
	ro_note_release(od->evp_ctx);
#ifdef RO_NEGATIVE_CONTROL
	/* The control withholds only the second free. The recursive Forget, the
	 * error it raises and the second release all still happen. */
	if (ro_release_count(od->evp_ctx) == 1)
		ro_free_probe(od->evp_ctx);
#else
	ro_free_probe(od->evp_ctx);
#endif

	/* The defect: od->owner was never cleared, so this runs during release
	 * and real resowner.c refuses it with an ERROR. */
	if (od->owner != NULL)
		ResourceOwnerForget(od->owner, PointerGetDatum(od),
							&osslcipher_resowner_desc);
	pfree(od);					/* not reached at the pin: Forget raised */
}

static void
ResOwnerReleaseOSSLCipher(Datum res)
{
	/* 17.5, verbatim in shape: no `cipher->owner = NULL;` before the free.
	 * 2aa6be6e64 adds exactly that line. */
	free_openssl_cipher((OSSLCipher *) DatumGetPointer(res));
}

RO_CASE(1) {
	ResourceOwner owner;
	OSSLCipher *od;
	unsigned	errors;

	owner = ResourceOwnerCreate(NULL, "corpus owner");

	od = MemoryContextAllocZero(ro_root, sizeof(OSSLCipher));
	od->evp_ctx = malloc(64);	/* EVP_CIPHER_CTX_new() */
	CHECK(od->evp_ctx != NULL, 10);

	ResourceOwnerEnlarge(owner);
	ResourceOwnerRemember(owner, PointerGetDatum(od),
						  &osslcipher_resowner_desc);
	od->owner = owner;

	ro_mark();
	errors = ro_release_with_abort_cleanup(owner);

	fprintf(stderr, "  releases of evp_ctx %p: %u (errors escaped: %u)\n",
			od->evp_ctx, ro_release_count(od->evp_ctx), errors);

	/* Both halves of the chain, each checked: the recursive Forget raised
	 * (errors >= 1) and the handle was freed twice (count == 2). */
	CHECK(errors >= 1, 11);
	CHECK(ro_release_count(od->evp_ctx) == 2, 12);
}
