/* cffi-lua #57 — Lua callback cdata ⟷ C closure_data use-after-free.
 * Source: ../../cffi-lua-57/boundary.md. ASan: heap-use-after-free READ size 4,
 * 32 bytes inside a 56-byte closure_data freed by destroy_closure.
 *
 * Two allocations: the callback cdata (cffi.cast) and the closure_data struct
 * (holds a libffi ffi_closure + a registry ref).
 *   Free-site (ffi.cc:127): callback:free() -> cdata_meta::cb_free
 *     (ffilib.cc:268) -> ffi::destroy_closure -> delete[] frees the 56-byte block.
 *   Stale-use (ffilib.cc:281): callback:set(fn) -> cdata_meta::cb_set reads
 *     fd.cd->fref on the freed block.
 * READ size 4 at OFFSET 32 (the ->fref field) -> interior address via cincoffset
 * on the revoked capability (assert-on-untagged FAULT route). Control: the read
 * returns and the row reports MISS.
 *
 * CAVEAT (from the case): the coupled object is FFI bridge plumbing, not a
 * third-party library resource — the weakest cross-language pair in the corpus,
 * carried for completeness. The memory event modeled here is identical to the
 * clean cases.
 */
#include "luac_shim.h"
#include <stdint.h>

#define CLOSURE_DATA_BYTES 56
#define FREF_OFF 32 /* the ->fref field ASan names */

static volatile uint64_t sink;

int main(void) {
  unsigned char *cd = (unsigned char *)malloc(CLOSURE_DATA_BYTES); /* new[] */
  if (!cd)
    abort();
  memset(cd, 0, CLOSURE_DATA_BYTES);

  unsigned char *fd_cd = cd; /* the callback cdata's cached closure_data* */

  free(cd); /* callback:free -> destroy_closure -> delete[] -> REVOKE */

  /* callback:set(fn) -> cb_set reads ->fref at offset 32. */
  sink = *(volatile uint32_t *)(fd_cd + FREF_OFF); /* ffilib.cc:281 */

  mock_report("luac_ffi_closure_uaf", "use-after-free-survived");
  return 0;
}
