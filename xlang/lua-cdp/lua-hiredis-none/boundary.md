# Boundary annotation — lua-hiredis (**no crossing object exists**)

This file documents why the CDP boundary the corpus looks for is **absent** in
the Lua-hiredis binding, i.e. why this is BLOCKED rather than a case.

### The object that would cross the boundary — and does not
The candidate object is a `redisReply*` (or a nested `reply->element[i]`), the C
struct hiredis returns from `redisCommandArgv`. For a CDP UAF it would have to be
handed to Lua as a **live handle** (userdata storing the pointer) while C retains
ownership of the freeing.

That handoff never happens. `push_reply` (`src/lua-hiredis.c:266`) converts the
reply to **pure Lua values** by type:
- `REDIS_REPLY_STRING`/`STATUS`/`ERROR` → `lua_pushlstring` (copy of `pReply->str`)
- `REDIS_REPLY_INTEGER` → `lua_pushinteger`
- `REDIS_REPLY_ARRAY` → `lua_createtable` + recurse over `pReply->element[i]` (`:322`)

No `lua_newuserdata` wraps a reply; no reply pointer is stored. The Lua value
returned owns **no** C memory.

### Owner vs. borrower
- **C owns the `redisReply*` for its whole (sub-second) lifetime**; `lconn_command`
  (`:340`) frees it with `freeReplyObject(pReply)` (`:364`) in the same call, after
  the copy. There is no borrower on the Lua side — Lua got a copy, not a pointer.
- This is *not even* the excluded "borrowed pointer into a Lua string" shape:
  the direction is reversed (C copies *into* Lua and frees its own buffer), so
  there is neither a two-object coupling nor a borrowed interior pointer.

### The other native resource: `redisContext*` (userdata-wrapped, but guarded)
The connection userdata (`:184`) does wrap `redisContext*`. Free-site:
`lconn_close` (`:409`) → `redisFree` under `if (pConn->pContext != NULL)` then
`pConn->pContext = NULL`. `__gc` is the *same* function (`#define lconn_gc
lconn_close`, `:424`). So the free is idempotent (second call sees NULL), and any
stale use is rejected by `check_connection` (`:189`, errors on NULL context).
No aliasing second handle exists. → no double-free, no UAF.

### The lifetime rule — not violated
"A Lua handle that references a foreign object must keep it alive for the handle's
lifetime." Here no Lua handle references a `redisReply*` at all (it holds a copy),
and the one handle over `redisContext*` frees exactly once under a null guard.
The rule has nothing to bite on.

### Capability note (revoke-on-free)
There is no derived capability to revoke: the reply is copied, not aliased, and
the context capability is dropped (`pContext=NULL`) at free. A revoke-on-free
model changes nothing observable here — which is the whole reason this is BLOCKED
and not a case.
