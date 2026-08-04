# lua-hiredis — **BLOCKED**: no filed hard-tier Lua↔C CDP UAF exists

**One line.** The canonical Lua binding to hiredis (`agladysh/lua-hiredis`) and
every C fork of it are **CDP-safe by construction**: a `redisReply*` is fully
**deep-copied** into pure Lua values and `freeReplyObject`'d inside the same C
call, so it never crosses the GC boundary as a handle; the only wrapped native
resource, the `redisContext*`, has a null-guarded `__gc`==`close`. There is no
`redisReply`/`redisContext` UAF or double-free filed anywhere in the ecosystem.

## Identity

| | |
|---|---|
| Target library | [`agladysh/lua-hiredis`](https://github.com/agladysh/lua-hiredis) (the canonical Lua↔hiredis C binding) |
| Language pair | **C ⟷ Lua** (reference Lua 5.4) |
| Native dep | hiredis 1.2.0 + redis-server 8.0.5 (both installed) |
| Commit surveyed | `5df62990fdec196dc88a1037aa809afd1f57f1e4` (HEAD, builds on Lua 5.4) |
| **Status** | **BLOCKED — no clean filed hard-tier CDP bug** (slot free to reassign) |

## Why there is no hard-tier CDP UAF (the search result)

Hard-tier CDP requires a Lua-GC userdata wrapping a **separate** native C
resource, one freed while the other still references it. Neither of the two
native resources hiredis exposes is coupled that way here:

**1. `redisReply*` — never crosses the boundary.**
`push_reply` (`src/lua-hiredis.c:266`) *materialises* the reply into pure Lua
values — `lua_pushlstring`/`lua_pushinteger`/`lua_createtable`, recursing over
`pReply->element[i]` (line 322) — and the caller `lconn_command`
(`:359`) does `nret = push_reply(...); freeReplyObject(pReply);` (`:364`)
in the same C function. The reply and every nested element are Lua copies; the
`redisReply*` is freed before control returns to Lua. **No Lua handle ever holds
`redisReply*` or `reply->element[i]`**, so the "nested element held after the
parent is freed" shape is structurally impossible. (Empirically confirmed — see
`evidence.txt`: an `LRANGE` reply comes back as a Lua `table` that survives two
full GCs + 5000-alloc heap churn with zero ASan reports.)

**2. `redisContext*` — held by a userdata, but double-free-guarded.**
The connection userdata wraps `redisContext*` (`:184`). `__gc` is literally
`#define lconn_gc lconn_close` (`:424`), and `lconn_close` (`:409`) frees only
when non-NULL and nulls the field: `if (pConn && pConn->pContext != NULL) {
redisFree(...); pConn->pContext = NULL; }`. Explicit `close()` followed by GC is
therefore a no-op the second time — no double-free — and `check_connection`
(`:189`) rejects any use after close. No second Lua handle aliases the same
context.

## Ecosystem survey (real search, 2026-08-04)

- **`agladysh/lua-hiredis`** issues #1 (TYPE command), #2 (nested-bulk parse
  depth limit), #4 (Lua 5.2 support) — **none is a UAF/double-free**. PRs #3/#5/#6/#7
  and the full commit log are Lua-version compatibility + packaging only; **no
  commit fixes a UAF or double-free**.
- **C forks** — `nmreis/lua-hiredis` (Lua 5.2), `zs-soft/lua-hiredis-cluster` —
  carry the **identical** deep-copy `push_reply`+immediate-`freeReplyObject`
  design (verified against their `src/lua-hiredis.c`). Empty issue trackers.
  The ~30 network forks are 2014 mass-mirrors; none diverges to a reply-userdata
  design.
- **`miketang84/bamboo-redis`** — pure-Lua wrapper (`bamboo-redis.lua`), no C
  boundary.
- **FFI bindings** (`koolhazz/hiredis_ffi`) — LuaJIT-FFI, out of scope for the
  PUC-Lua C-module toolchain; no filed CDP UAF.

## Verdict

**BLOCKED.** Following the task's own instruction ("If lua-hiredis has NO clean
filed hard-tier CDP bug after real searching, say so clearly and mark the case
dir BLOCKED … do NOT force a non-CDP or reconstructed bug"): the canonical
binding is safe *by construction* for both native resources, and no UAF/double-free
is filed as an issue, PR, or commit in the Lua-hiredis ecosystem. Nothing to
reproduce. Reassign the slot.

## PASS signature (of this BLOCKED case)

`./build.sh && ./run.sh` builds the real binding against the toolchain and runs
the strongest attempt at the described CDP shape under ASan. The run prints
`OK-reply-is-a-lua-copy` and `OK-context-not-double-freed` with **no ASan report**
(the empirical proof of non-vulnerability), then exits **3 = BLOCKED** with the
reason — so the pipeline is deliberately non-zero and never a false PASS.
