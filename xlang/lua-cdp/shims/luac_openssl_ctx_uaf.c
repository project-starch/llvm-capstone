/* lua-openssl #141 — Lua userdata ⟷ C EVP_CIPHER_CTX cross-domain double-free.
 *
 * Source of truth: ../../lua-openssl-141/{CASE.md,boundary.md,evidence.txt}.
 * Our own ASan reproduction: heap-use-after-free, READ of size 8 at offset 0 of
 * a 168-byte EVP_CIPHER_CTX, in EVP_CIPHER_CTX_reset (evp_enc.c:26) reached from
 * openssl_cipher_ctx_free (cipher.c:551), run by the userdata __gc (GCTM); the
 * block was freed by the SAME function at cipher.c:552 during c:close().
 *
 * The two distinct allocations (why this is CDP, not a borrow into one object):
 *   1. the Lua full userdata that boxes the pointer (PUSH_OBJECT) — the GC handle
 *   2. a separate 168-byte EVP_CIPHER_CTX from libcrypto (EVP_CIPHER_CTX_new)
 *
 * The defect, verbatim from cipher.c:546-554: openssl_cipher_ctx_free reads the
 * boxed pointer (CHECK_OBJECT), calls EVP_CIPHER_CTX_cleanup(ctx) — a READ of
 * the ctx — then EVP_CIPHER_CTX_free(ctx), and does NOT null the box. c:close()
 * and __gc are both registered to this one function, so the box outlives the
 * free and __gc re-enters with a stale pointer.
 *
 * What Capstone must do: c:close()'s free REVOKES the 168-byte block. __gc then
 * loads the (now-revoked) boxed pointer and dereferences it at offset 0 — the
 * cipher.c:551 read. A plain load through a revoked capability, no arithmetic,
 * so this takes the DELIVERY route (clean cause-25 "Cap mem access on revoked
 * capability"), not the cincoffset assert gap. Control (no-revoke): the read
 * returns stale bytes and the row reports MISS.
 */
#include "luac_shim.h"
#include <stdint.h>

#define CTX_BYTES 168 /* the EVP_CIPHER_CTX region ASan names */

/* The Lua full userdata: a box holding the C pointer (PUSH_OBJECT). Distinct
 * allocation from the ctx it points at. Lives on the domain stack here; only the
 * ctx block is heap-allocated and revocable, exactly as in the real bug where
 * the box is never nulled. */
struct cipher_userdata {
  void *ctx;
};

static volatile uint64_t sink;

/* openssl.cipher.decrypt_new -> EVP_CIPHER_CTX_new -> CRYPTO_zalloc. */
static void *evp_cipher_ctx_new(void) {
  void *p = malloc(CTX_BYTES);
  if (!p)
    abort();
  memset(p, 0, CTX_BYTES);
  return p;
}

/* openssl_cipher_ctx_free (cipher.c:546-554), the VULNERABLE tree (0017afa):
 *   read boxed ptr (CHECK_OBJECT) -> cleanup READS ctx (551) -> free ctx (552),
 *   box NOT nulled. First call (close): ctx live, read harmless, free revokes.
 *   Second call (__gc): ctx stale, the 551 read is the heap-use-after-free. */
static void openssl_cipher_ctx_free(struct cipher_userdata *ud) {
  void *ctx = ud->ctx;                       /* CHECK_OBJECT reads the box */
  sink = *(volatile uint64_t *)ctx;          /* cipher.c:551 cleanup READ, off 0 */
  free(ctx);                                 /* cipher.c:552 free -> REVOKE */
  /* vulnerable: the boxed pointer is left dangling (fix adds FREE_OBJECT) */
}

int main(void) {
  struct cipher_userdata ud;
  ud.ctx = evp_cipher_ctx_new();

  openssl_cipher_ctx_free(&ud); /* c:close()  — crossing 1: frees + revokes ctx */
  /* ... collectgarbage('collect') ... */
  openssl_cipher_ctx_free(&ud); /* __gc (GCTM) — crossing 2: reads freed ctx    */

  mock_report("luac_openssl_ctx_uaf", "use-after-free-survived");
  return 0;
}
