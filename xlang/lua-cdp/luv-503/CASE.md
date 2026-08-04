# luv #503 — long-lived `lua_State*` (a GC'd coroutine) ⟷ luv C handle context use-after-free

**One line.** `require("luv")` from a coroutine makes luv store *that coroutine's*
`lua_State` in its per-loop context (`ctx->L`); when the coroutine is GC'd, every
later luv handle callback dereferences the freed `lua_State`.

## Identity

| | |
|---|---|
| Library | [`luv`](https://github.com/luvit/luv) (luvit) |
| Language pair | **C ⟷ Lua** (reference Lua 5.4; issue also affects 5.2+/LuaJIT) |
| Upstream | https://github.com/luvit/luv/issues/503 (fix PR #734) |
| Vulnerable commit | **`e2d3d18`** (parent of the fix; `luaopen_luv` does `ctx->L = L`, luv.c:865) |
| Fix commit | **`ba4589c`** ("Use main thread of current Lua state for callbacks, when known") |
| Native dep | libuv (verified 1.51.0) |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC object:** the coroutine's `lua_State` (a Lua thread), created by
   `coroutine.wrap`/`create`, freed by the Lua GC (`luaM_free_`, 208-byte block).
2. **Separate native resource:** luv's C per-loop context `luv_ctx_t`, held in the
   registry, whose `ctx->L` field is a **raw C pointer** to that coroutine —
   invisible to the GC, so nothing pins the thread.

**Direction:** GC-frees. The coroutine `lua_State` is collected in the main thread
while `ctx->L` still points at it; the next luv handle callback
(`luv_close_cb`/`luv_timer_cb` → `luv_call_callback`/`luv_unref_handle`) runs
`lua_settop`/`luaL_unref`/`lua_rawgeti` on the freed thread.

## Root cause

`luaopen_luv` runs on whatever Lua thread first `require`s luv. In the vuln tree it
unconditionally stores that thread: `ctx->L = L` (luv.c:865). Loaded from a
coroutine, `ctx->L` is the coroutine — which "may become suspended (by calling
yield)" or die and be collected. luv uses `ctx->L` for *all* callbacks
(`luv_close_cb` reads `L = data->ctx->L`, handle.c:97). The fix resolves and stores
the **main thread** (`LUA_RIDX_MAINTHREAD`, Lua 5.2+), which can never yield and is
never collected.

## Reproduction status

**REPRODUCED (2026-08-04), with control.**

- Env: PUC Lua 5.4.7 (shared toolchain), libuv 1.51.0 (shared), gcc 15, luv cmake
  module build (`-DWITH_SHARED_LIBUV=ON -DBUILD_MODULE=ON`), plain `-g -O0`.
- Detection: **valgrind** memcheck (see note below), reduced from the issue's
  snippet to `trigger.lua`: require luv in a coroutine, keep the timer in a main
  var, `collectgarbage()` the coroutine (dangling `ctx->L`), then `timer:close()`
  + `uv.run()` to drive a luv callback onto the freed thread.
- Vuln `e2d3d18`: **valgrind Invalid read** of the freed 208-byte `lua_State` at
  `lua_settop` ← `luv_call_callback` (lhandle.c:91) ← `luv_close_cb` (handle.c:99),
  and `luaL_unref` ← `luv_unref_handle` (lhandle.c:106) ← `luv_close_cb`.
- Control, fixed `ba4589c`: clean, `DONE`, 0 valgrind errors.
- `./build.sh && ./run.sh` → PASS.

### Why valgrind, not ASan (deliberate deviation from the luv-696 recipe)

Building luv.so with ASan and preloading libasan (the luv-696 approach) does **not**
observe this UAF: the faulting dereference happens *inside* `liblua5.4.so`
(`lua_settop`/`luaL_unref`), which is not ASan-instrumented, and ASan's quarantine
preserves the freed bytes so the stale read silently "succeeds" (prints `DONE` on
both trees — verified). valgrind instruments liblua too and flags the read; issue
#503 was itself diagnosed under valgrind. `build.sh` therefore builds plain
(non-ASan) binaries (ASan is also incompatible with a valgrind run). Upgrade path:
an ASan-built `liblua5.4` would let ASan catch it, but the shared toolchain Lua is
prebuilt read-only.

The direct/verbatim issue snippet (create a timer in a coroutine, drop all refs,
`collectgarbage`) does **not** fire a callback on this Lua 5.4.7 build — the timer
stays registry-pinned by luv and `package.loaded` pins the loop, so the only
dangling-`ctx->L` deref would be at `lua_close` via exit-time GC ordering, which
happens not to trip here. `trigger.lua` forces the *same* root-cause deref
deterministically (analogous to luv-696's reduced trigger).

## PASS signature

Vuln: valgrind `Invalid read` of a freed 208-byte block (the coroutine
`lua_State`) with a `luv_close_cb`/`luv_gc_cb` frame reached via
`lua_settop`/`luaL_unref`/`lua_rawgeti`. Fixed: `DONE`, no valgrind errors. Both
required.
