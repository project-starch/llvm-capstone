# Boundary annotation — LuaDBI #35

### The object that crosses the boundary

A raw `PGconn *`, allocated by libpq (`makeEmptyPGconn` via `PQsetdbLogin`) and
owned by the *connection* full userdata. When `db:prepare(...)` builds a
*statement* userdata, the vulnerable driver copies that same pointer into the
statement's own struct (`dbd_postgresql.h`: `PGconn *postgresql;`). The C pointer
is what crosses — cached in a second Lua object with an independent lifetime.

### Owner vs. borrower

- **libpq (native) owns the memory.** `PQsetdbLogin()` allocated the `PGconn`;
  `PQfinish()` frees it.
- **Lua (managed) owns the two handle lifetimes.** The connection userdata and
  the statement userdata are collected/closed independently by the GC and by
  explicit `:close()`.
- The bug: the statement caches a **raw copy** of the connection's native handle,
  so closing the connection frees a `PGconn` the statement still points at. The
  statement is the borrower; it never re-validates or nulls its copy.

### Free site (first crossing)

`db:close()` → `connection_close` → `PQfinish(conn->postgresql)`
(`dbd/postgresql/connection.c:141`). libpq's `freePGconn` (`fe-connect.c:4468`)
`free()`s the 1056-byte `PGconn`. `connection_close` nulls **the connection's**
field (`conn->postgresql = NULL`) but not the statement's cached copy.

### Stale-use site (second crossing)

`stmt:execute()` → `statement_execute` reads the cached handle at
`dbd/postgresql/statement.c:145`:

```c
/* Sanity check - is database still connected? */
if (PQstatus(statement->postgresql) != CONNECTION_OK)   /* reads the FREED PGconn */
```

`PQstatus` (`fe-connect.c:7196`) dereferences the freed block → ASan
`heap-use-after-free` (READ of size 4). Ironically the line is a *sanity check*
added by commit `d5b0bfb` — but it validates the stale copy by reading it, so the
check itself is the use-after-free.

### The lifetime rule that is violated

A child handle (statement) must not outlive-and-dereference a native resource
owned by its parent (connection). Either it holds no independent pointer and
always indirects through the live parent object, or the parent's free must
invalidate every cached copy. The fix (`f562ccd`) takes the first route: the
statement stores `connection_t *conn` and reads `conn->postgresql` live, which is
NULL after close, so `PQstatus(NULL)` is a clean `CONNECTION_BAD` instead of a
dereference of freed memory.

### Capability note (revoke-on-free)

On a revoke-on-free allocator the first free (`db:close()` → `PQfinish`)
**revokes** the capability to the `PGconn` block. The statement's cached copy is
then a revoked capability: `PQstatus`'s load through it — capability arithmetic /
dereference of a revoked cap — faults at the contract point instead of reading a
recycled `PGconn`. The interesting property here is that the stale pointer lives
in a *different* Lua object than the one that freed it, so a scheme keying
revocation to the owning object still has to cover the aliased copy.
