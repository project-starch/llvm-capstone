# Boundary annotation — luv #503

### The object that crosses the boundary
A Lua coroutine's `lua_State*` (the running thread that first `require`d luv),
captured by C as `ctx->L` inside luv's per-loop context `luv_ctx_t` (held in the
Lua registry) at `luaopen_luv` (luv.c:865, `ctx->L = L`).

### Owner vs. borrower
- **Lua (managed) owns the coroutine's lifetime.** The `lua_State` is an ordinary
  GC object; once the coroutine is unreferenced it is collected (`luaM_free_`).
- **luv's C context borrows it as a raw pointer.** `ctx->L` is not a GC ref and
  does not pin the thread, yet luv treats it as long-lived and uses it for every
  handle callback.

### Free site
Main thread: the coroutine becomes unreachable and `collectgarbage()` frees its
`lua_State` (208-byte block) via the Lua GC → `luaM_free_`. `ctx->L` now dangles.

### Stale-use site (one crossing away)
Any subsequent luv handle callback: `luv_close_cb` (handle.c:97) does
`L = data->ctx->L`, then `luv_call_callback` (lhandle.c:91, `lua_settop`) and
`luv_unref_handle` (lhandle.c:106, `luaL_unref`) operate on the freed `lua_State`
→ Invalid read. (`luv_timer_cb`, timer.c:39, is the same bug via `lua_rawgeti`.)

### The lifetime rule that is violated
A `lua_State*` retained across callbacks must be one that outlives them and is
always resumable — i.e. the **main thread**, which never yields and is never
collected. Storing the loading coroutine's thread (`ctx->L = L`) captures a state
that can suspend or be freed. `ba4589c` stores `LUA_RIDX_MAINTHREAD` instead.

### Capability note (revoke-on-free)
Revoke-on-free revokes the coroutine `lua_State` capability when the GC frees the
thread; luv's stored `ctx->L` copy is revoked too, so the first callback faults at
the boundary (`lua_settop` on a revoked capability) instead of reading a freed
Lua thread and corrupting interpreter state.
