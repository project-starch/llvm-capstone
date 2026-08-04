/* lmdb value-after-txn — Lua val userdata ⟷ LMDB overflow-page buffer UAF.
 * Source: ../../lmdb-value-after-txn/boundary.md. ASan: heap-use-after-free READ
 * on a freed multi-page overflow buffer. DOCUMENTED-CONTRACT reproduction (a
 * deliberately-unsafe minimal binding, minilmdb.c) — NOT an upstream-filed bug.
 *
 * Two allocations: the minilmdb.val userdata (stores only a BORROWED
 * {const char* data; size_t size} into the txn page, not a copy) and the LMDB
 * overflow-page buffer holding the value bytes (a multi-page value so LMDB
 * free()s it instead of pooling).
 *   Free-site (minilmdb.c:101): txn:commit() -> l_commit -> mdb_txn_commit ->
 *     free of the dirty overflow buffer. (txn:abort() is identical.)
 *   Stale-use (minilmdb.c:117): val:read() -> l_val_read dereferences the
 *     borrowed v->data[i] -> read of the freed overflow page.
 * READ at OFFSET 0 (the value head) -> plain load through the revoked capability
 * (clean cause-25 route). Control: the read returns; row reports MISS.
 *
 * CAVEAT (from the case): borrowed-pointer / constructed shape — the val is a
 * borrow into the page rather than a wrap-a-pointer pair. Carried as a
 * documented-contract case, not a filed bug.
 */
#include "luac_shim.h"
#include <stdint.h>

#define OVERFLOW_BYTES 8192 /* a multi-page value: LMDB free()s, does not pool */

static volatile uint64_t sink;

int main(void) {
  unsigned char *page = (unsigned char *)malloc(OVERFLOW_BYTES); /* mdb_put buffer */
  if (!page)
    abort();
  memset(page, 0, OVERFLOW_BYTES);

  /* txn:get(k) stores only a borrowed pointer into the page, no copy. */
  unsigned char *val_data = page;

  free(page); /* txn:commit -> mdb_txn_commit -> free -> REVOKE */

  /* val:read -> l_val_read derefs the borrowed v->data at the value head. */
  sink = *(volatile uint64_t *)val_data; /* minilmdb.c:117 */

  mock_report("luac_lmdb_value_uaf", "use-after-free-survived");
  return 0;
}
