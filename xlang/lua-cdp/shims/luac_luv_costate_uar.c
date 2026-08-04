/* luv #503 — luv C context ⟷ coroutine lua_State back-pointer use-after-free.
 * Source: ../../luv-503/boundary.md. valgrind: invalid read, 24 bytes inside a
 * 208-byte lua_State freed by the Lua GC (luaM_free_).
 *
 * Two allocations: the coroutine's lua_State (a Lua thread) and luv's C per-loop
 * context luv_ctx_t, which caches a raw back-pointer ctx->L into that thread.
 *   Free-site: main thread: the coroutine becomes unreachable and
 *     collectgarbage() frees its 208-byte lua_State via luaM_free_; ctx->L dangles.
 *   Stale-use (handle.c:97): a later luv handle callback: luv_close_cb does
 *     L = data->ctx->L, then luv_call_callback (lua_settop) operates on the freed
 *     lua_State's interior fields.
 * READ at OFFSET 24 -> interior address via cincoffset on the revoked capability
 * (assert-on-untagged FAULT route). Control: the read returns; row reports MISS.
 *
 * CAVEAT (spare-tier, from the case): the freed block is Lua-managed memory (a
 * lua_State), not a native heap object — directly analogous to lua-curl-80. The
 * allocator-visible event (alloc -> free -> deref of a cached back-pointer) is
 * the same as the native-resource cases; Lua's l_alloc revokes on free.
 */
#include "luac_shim.h"
#include <stdint.h>

#define LUA_STATE_BYTES 208
#define THREAD_FIELD_OFF 24 /* the lua_State field valgrind names */

static volatile uint64_t sink;

int main(void) {
  unsigned char *co_L = (unsigned char *)malloc(LUA_STATE_BYTES); /* new coroutine */
  if (!co_L)
    abort();
  memset(co_L, 0, LUA_STATE_BYTES);

  unsigned char *ctx_L = co_L; /* luv_ctx_t caches ctx->L -> the coroutine */

  free(co_L); /* collectgarbage() -> luaM_free_ frees the lua_State -> REVOKE */

  /* luv_close_cb: L = ctx->L; lua_settop(L,...) touches a field at offset 24. */
  sink = *(volatile uint64_t *)(ctx_L + THREAD_FIELD_OFF); /* handle.c:97 -> lua_settop */

  mock_report("luac_luv_costate_uar", "use-after-free-survived");
  return 0;
}
