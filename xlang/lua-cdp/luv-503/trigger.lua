-- luv #503 — deterministic reproducer.
--
-- Root cause (vuln e2d3d18, luv.c:865 `ctx->L = L`): when luv is require()d from
-- a coroutine, luv stores THAT coroutine's lua_State in ctx->L and uses it for
-- every handle callback. When the coroutine is collected, ctx->L dangles; the
-- next luv callback derefs the freed lua_State.
--
-- The issue's verbatim snippet only trips this at lua_close via exit-time GC
-- ordering, which does not fire a callback on this Lua 5.4.7 build (the timer
-- stays registry-pinned by luv, and package.loaded pins the loop). So we force
-- the SAME dangling-ctx->L deref deterministically: keep the timer in a main
-- variable, let the coroutine die and be GC'd, then drive a luv callback
-- (timer:close -> luv_close_cb) which reads the freed coroutine lua_State.
local uv, timer
coroutine.wrap(function ()
  uv = require "luv"          -- luaopen_luv runs on THIS coroutine -> ctx->L = coroutine
  timer = uv.new_timer()      -- kept alive via main-thread `timer`
  coroutine.yield()           -- suspend, then become unreferenced
end)()
for i = 1, 5 do collectgarbage("collect") end  -- free the coroutine; ctx->L now dangling
timer:close()                 -- luv_close_cb reads ctx->L (freed) -> lua_settop/luaL_unref UAF
uv.run()                      -- process the close callback
print("DONE")
