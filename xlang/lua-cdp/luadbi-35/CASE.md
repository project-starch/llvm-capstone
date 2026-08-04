# LuaDBI #35 — Lua statement userdata ⟷ C `PGconn` use-after-free

**One line.** A Lua statement userdata caches its own raw copy of the
connection's libpq `PGconn*`; `db:close()` frees that `PGconn` (via `PQfinish`),
but the statement keeps the stale pointer, so `stmt:execute()` dereferences the
freed handle — a cross-domain use-after-free.

## Identity

| | |
|---|---|
| Library | [`LuaDBI`](https://github.com/mwild1/luadbi) (mwild1) — PostgreSQL driver (`dbd/postgresql/`) |
| Language pair | **C ⟷ Lua** (reference Lua 5.1–5.4) |
| Upstream | https://github.com/mwild1/luadbi/issues/35 — "Segfault in SQLite3 driver" (filed 2016-03-03, closed 2016-03-04). Maintainer: *"Fixed in all open-source drivers."* |
| CVE / GHSA | none assigned |
| Native library | PostgreSQL 16.4 `libpq` (built from source; the client's copy is ASan-instrumented — see below) |
| Vulnerable commit | **`b1075359ab3c50faa480775a7e882912b3186166`** (= `f562ccd~1`) — statement stores a raw `PGconn *postgresql;` |
| Fix commit | **`f562ccdc93c4068db3350d76ed6baf0848c51aaf`** — "Make the statement object refer to the connection object, instead of the connection handle, as in the MySQL driver." (drops the cached raw handle; reads `statement->conn->postgresql` live). |

**On the filing (honest provenance).** Issue #35's repro was written against the
SQLite3 driver, and the maintainer fixed it "in all open-source drivers." The two
drivers differ in *shape*: the SQLite3 statement stores the connection *struct*
and reads `conn->sqlite` (nulled on close) → a use-after-close **NULL-deref**; the
**PostgreSQL** statement stores a *raw copy of the native handle*, so the same
close→execute sequence is a genuine **heap-use-after-free of the freed `PGconn`**.
We reproduce the PostgreSQL variant because it is the true CDP UAF. Its fix landed
in two steps — `d5b0bfb` ("Fix Github bug #35 in the Postgresql driver") added a
`PQstatus()` "sanity check" that *still reads the stale copy*, and `f562ccd`
removed the raw copy entirely. The clean differential is therefore `f562ccd~1`
(the stale read is live, inside the d5b0bfb guard) → `f562ccd` (structural fix).

## The two coupled objects (why this is unambiguous CDP)

1. **Lua-GC handle:** the statement full userdata returned by `db:prepare(...)`
   (`lua_newuserdata` in `dbd_postgresql_statement_create`, `statement.c`).
2. **Separate native resource:** a 1056-byte libpq `PGconn`, allocated by
   `makeEmptyPGconn` (via `PQsetdbLogin`), owned by the *connection* userdata.

Two distinct allocations. At the vulnerable commit the statement stores its own
copy of the pointer — `dbd_postgresql.h`: `PGconn *postgresql;`, set by
`statement->postgresql = conn->postgresql;` (`statement.c`, `statement_create`).
This is **not** the excluded "raw pointer into a Lua string" shape — the freed
object is a libpq C struct, not Lua-VM internals.

**Direction:** native-frees. `db:close()` frees the `PGconn` (crossing 1;
`connection_close` → `PQfinish`); `stmt:execute()` later reads the same, now-stale
pointer (crossing 2; `statement_execute` → `PQstatus`) → use-after-free.

## Dependencies

- **PostgreSQL 16.4, built from source** (`build.sh`) — this box ships no
  postgres server and no `libpq-fe.h`. One source build yields the server
  binaries (`initdb`/`postgres`/`pg_ctl`) that `run.sh` runs on a private unix
  socket, plus the client `libpq`. Same discipline as `lua-openssl-141` (OpenSSL
  from source); no apt, no docker.
- The shared reference **Lua 5.4.7** from `../_toolchain` (read-only).
- `libpq` is built **twice**: an ordinary copy in `lib/` for the server, and an
  **ASan copy in `lib-asan/`** for the client. This is required: the stale read is
  `PQstatus()`, which lives *inside* libpq, so with an un-instrumented libpq the
  read is never shadow-checked and ASan reports nothing (verified — it silently
  returns `CONNECTION_BAD`). Instrumenting the client's libpq turns it into the
  labelled heap-use-after-free with alloc/free/use stacks (exactly the split
  `lua-openssl-141` uses: ASan libcrypto, ordinary everything else).
- LuaDBI @2016 uses the removed name `luaL_checkint` at one connect-time site
  (the port argument, off the UAF path); `run.sh` force-injects a one-macro build
  shim via `gcc -include` (the analogue of `lua-openssl-141`'s `-DOPENSSL_NO_SM2`).

## Reproduction status

**REPRODUCED (2026-08-04), with control.**

- Env: PostgreSQL 16.4 (from source; ASan client libpq), PUC Lua 5.4.7 (shared
  toolchain), gcc 15.2 ASan.
- Vulnerable `b1075359`: ASan **heap-use-after-free**, `READ of size 4` in
  `PQstatus` (`fe-connect.c:7196`) from `statement_execute` (`statement.c:145`)
  run by `stmt:execute()`. The 1056-byte `PGconn` was **freed** by `PQfinish`
  (`fe-connect.c:4604`) ← `connection_close` (`connection.c:141`) during
  `db:close()`, and **allocated** by `makeEmptyPGconn` ← `PQsetdbLogin` ←
  `connection_new` (`connection.c:86`).
- Control, fixed `f562ccd`: **no ASan report** — trigger prints
  `NO-UAF ... Statement unavailable: database closed`.
- Full trace + control in `evidence.txt`.

## PASS signature

`run.sh` passes iff **both** halves of the differential hold:

- **Vulnerable `b1075359`:** ASan reports `heap-use-after-free`, with the read
  reached from the statement's execute method and the free from the connection's
  close:

  ```
  #0 PQstatus                fe-connect.c:7196
  #1 statement_execute       dbd/postgresql/statement.c:145   (stmt:execute())
  freed by: PQfinish <- connection_close  dbd/postgresql/connection.c:141  (db:close())
  ```

  Concretely `run.sh` requires `heap-use-after-free` **AND** `statement_execute`
  **AND** `PQfinish` **AND** `connection_close` in the output.

- **Fixed `f562ccd` (control):** no ASan report; the trigger runs to completion
  and prints `NO-UAF`.

Either half missing = FAIL. The first free is `db:close()` (`PQfinish`); the
stale use is `stmt:execute()` (`PQstatus` on the cached raw handle). The fix makes
the statement read the live `conn->postgresql` (NULL after close) instead.
