/* luaossl #124 — Lua store userdata ⟷ C X509_STORE co-owned double-free.
 * Source: ../../luaossl-124/boundary.md. ASan: heap-use-after-free WRITE size 4,
 * 136 bytes inside a 152-byte X509_STORE freed by X509_STORE_free.
 *
 * Two allocations: the x509.store userdata and the X509_STORE (X509_STORE_new),
 * co-owned by an SSL_CTX via set0 WITHOUT an up-ref (the ownership confusion).
 *   Free-site (x509_lu.c:230): the store userdata's __gc -> xs__gc ->
 *     X509_STORE_free, refcount 1->0 -> CRYPTO_free releases the 152-byte block.
 *   Stale-use (x509_lu.c:212): the SSL_CTX userdata's __gc -> sx__gc ->
 *     SSL_CTX_free -> X509_STORE_free(ctx->cert_store) -> CRYPTO_DOWN_REF
 *     reads-and-writes the refcount word of the already-freed store.
 * WRITE size 4 at OFFSET 136 (the refcount) -> interior store through the
 * revoked capability (assert-on-untagged FAULT route). Control: the refcount
 * decrement completes and the row reports MISS.
 */
#include "luac_shim.h"
#include <stdint.h>

#define X509_STORE_BYTES 152
#define REFCOUNT_OFF 136 /* the refcount word ASan names */

int main(void) {
  unsigned char *store = (unsigned char *)malloc(X509_STORE_BYTES); /* X509_STORE_new */
  if (!store)
    abort();
  memset(store, 0, X509_STORE_BYTES);
  *(volatile uint32_t *)(store + REFCOUNT_OFF) = 1; /* refcount = 1 (no up-ref) */

  /* SSL_CTX->cert_store aliases the same store (set0, no up-ref). */
  unsigned char *ctx_cert_store = store;

  free(store); /* xs__gc -> X509_STORE_free (rc 1->0) -> REVOKE */

  /* sx__gc -> SSL_CTX_free -> CRYPTO_DOWN_REF on the freed store's refcount. */
  *(volatile uint32_t *)(ctx_cert_store + REFCOUNT_OFF) -= 1; /* x509_lu.c:212 */

  mock_report("luac_x509_store_dblfree", "use-after-free-survived");
  return 0;
}
