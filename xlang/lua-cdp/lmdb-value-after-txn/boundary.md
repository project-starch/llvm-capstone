# Boundary annotation — LMDB value-after-txn UAF

### The object that crosses the boundary
The zero-copy pointer LMDB returns from `mdb_get`: `MDB_val.mv_data` points
**directly into the transaction's page** (LMDB never copies value bytes out).
`minilmdb.c`'s `txn:get(k)` stores that `{mv_data, mv_size}` in a
`minilmdb.val` Lua userdata **without copying** — the userdata borrows a raw
pointer into LMDB-txn-owned memory.

### Owner vs. borrower
- **The LMDB transaction/page domain owns the bytes.** The value lives in an
  overflow page buffer that `mdb_put`→`mdb_cursor_put` malloc'd inside the write
  txn; the txn frees it at teardown.
- **The `minilmdb.val` userdata (Lua GC) borrows** a bare pointer into that page,
  with no copy and no handle keeping the txn/page alive.

### Free site
`txn:commit()` → `l_commit` (minilmdb.c:101) → `mdb_txn_commit` → `free` of the
dirty overflow buffer (liblmdb). `txn:abort()` → `l_abort` (minilmdb.c:106) →
`mdb_txn_abort` frees it identically. "End of the transaction" per the contract.

### Stale-use site (one crossing later)
`val:read()` → `l_val_read` (minilmdb.c:117) dereferences the borrowed
`v->data[i]` — a read of the freed overflow page → ASan heap-use-after-free.

### The lifetime rule that is violated
lmdb.h:249-251 / :1275-1276 — "Values returned from the database are valid only
until a subsequent update operation, or the end of the transaction." A binding
that surfaces the value must either copy it out before the txn ends (what
lightningmdb does) or keep the owning txn alive for the handle's lifetime. This
binding does neither: the handle outlives the txn and reads freed memory.

### Capability note (revoke-on-free)
Revoke-on-free revokes the capability to the overflow-page block at the txn's
`free`. The `minilmdb.val` handle's derived capability is revoked with it, so
`val:read()` faults at the dereference — the same signal ASan gives here, but
enforced in hardware, and it would also catch the *small-value* case that ASan
misses today because LMDB pools (does not `free()`) single-page dirty buffers.
