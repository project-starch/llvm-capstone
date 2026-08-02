/* Row 7 — RUSTSEC-2022-0070 / GHSA-969w-q74q-9j8v.
 *
 * secp256k1's `preallocated_gen_new` had an incorrect lifetime bound, so the C
 * context it returns could outlive the Rust buffer it is constructed in. Every
 * later use reads the freed context -- reachable from entirely safe Rust.
 *
 * evidence.txt: valgrind "Invalid read of size 4" at offset 0 and "of size 8"
 * at offsets 40 and 48, all inside a 208-byte block free'd. The size-8 read at
 * offset 40 is modelled here: it is the first pointer-width field the C code
 * follows, and matches the width the other rows report.
 *
 * THIS IS THE CORPUS'S CLEANEST CROSS-DOMAIN LEND. Unlike the mruby rows, where
 * the stale pointer is engine-internal (mruby's C code caching a pointer into
 * mruby's own register stack), here the HOST owns the memory and lends it:
 *
 *     host   = Rust, allocates the 208-byte context storage
 *     lend   = preallocated_gen_new(&mut buf) -- one call, one buffer
 *     engine = libsecp256k1, keeps a pointer into that storage
 *     free   = Vec::drop on the host side, while the engine still holds it
 *
 * That distinction matters for a boundary-only revocation scheme: this row's
 * pointer genuinely crosses the domain line, so a scheme that protects only
 * lent pointers still catches it. Six of the mruby rows would not qualify.
 *
 * Plain malloc/free is the faithful model, as in every other row: the Rust Vec
 * is an ordinary heap allocation and Vec::drop is an ordinary free. That is
 * exactly the event a revocation mechanism observes.
 */
#include "../mock-mruby/mock_mruby.h" /* mock_report only */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CONTEXT_BYTES 208 /* valgrind: "a block of size 208 free'd" */
#define ACCESS_OFF 40     /* valgrind: "40 bytes inside" */

/* The storage the HOST allocates and lends. In the real row this is a
 * Vec<AlignedType> living on the Rust heap. */
typedef struct PreallocStorage {
  unsigned char opaque[CONTEXT_BYTES];
} PreallocStorage;

/* The context the ENGINE builds inside the host's storage. It is just a pointer
 * into that buffer -- which is the whole defect: the type system was supposed
 * to stop this outliving `storage`, and did not. */
typedef struct Secp256k1Context {
  const unsigned char *prealloc; /* borrowed, no lifetime binding it */
} Secp256k1Context;

static PreallocStorage *host_alloc_storage(void) {
  PreallocStorage *s = (PreallocStorage *)malloc(sizeof *s);
  if (!s)
    abort();
  memset(s, 0, sizeof *s);
  return s;
}

/* The lend point: secp256k1_context_preallocated_create. The engine records a
 * pointer into host memory and returns a handle carrying it. */
static Secp256k1Context context_preallocated_create(const PreallocStorage *s) {
  Secp256k1Context ctx;
  ctx.prealloc = (const unsigned char *)s;
  return ctx;
}

/* PublicKey::from_secret_key -> secp256k1_ec_pubkey_create, which consults the
 * preallocated generator tables. This is the stale dereference. sign_ecdsa does
 * NOT touch them, which is why the native reproducer uses from_secret_key. */
static uint64_t ecmult_gen(const Secp256k1Context *ctx) {
  uint64_t table_entry;
  memcpy(&table_entry, ctx->prealloc + ACCESS_OFF, sizeof table_entry);
  return table_entry; /* READ of size 8 */
}

static volatile uint64_t sink;

int main(void) {
  PreallocStorage *storage = host_alloc_storage();

  /* preallocated_gen_new(&mut buf) -- the context escapes on a bogus 'static. */
  Secp256k1Context ctx = context_preallocated_create(storage);

  /* Vec::drop at the end of the enclosing scope. The host frees the storage
   * while the engine's context still points into it. */
  free(storage);

  /* Any later use of the context reads the freed block. */
  sink = ecmult_gen(&ctx);

  mock_report("secp256k1_preallocated_uaf", "use-after-free-survived");
  return 0;
}
