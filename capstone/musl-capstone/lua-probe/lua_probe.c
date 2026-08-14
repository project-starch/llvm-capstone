/* Reference Lua 5.4, compiled against musl, running inside a pure-capability
 * domain.
 *
 * The chunk is the same one the existing cjalr bring-up uses
 * (history/06-08-2026_19-45-00_lua-runs-on-capstone-cjalr-jumptables.md), so a
 * result here is directly comparable with the hand-written-libc build: build a
 * table of 20 squares and return the last one. 400 is the answer, and getting it
 * requires the parser, the VM, the GC and a realloc that MOVES the table's array
 * part while keeping its contents.
 *
 * WHAT THIS REPLACES. The existing Lua domain build carries 1008 lines of
 * hand-written libc (`capstone-lua/capstone_lua_libc.h` + `lua_libc.c`). Here
 * the same interpreter is built against musl instead.
 *
 * Loaded from a BUFFER, not a file: luaL_loadfilex needs fopen, and musl's
 * fopen.c is one of the files that does not compile yet. See lua_probe_stubs.c.
 */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

extern long __capstone_hc_write(long fd, const char *buf, unsigned long count);

#define SAY(s) __capstone_hc_write(1, (s), sizeof(s) - 1)

/* See musl-hello/musl_hello.c: padding buys dom_data, which is where the stack
   lives, and the yield frame alone is 256 bytes. Lua needs considerably more
   than the hello program, so this is sized up rather than merely non-degenerate.
   `retain` plus the live reference below, or --gc-sections drops it. */
#ifndef LUA_PROBE_PAD_WORDS
#define LUA_PROBE_PAD_WORDS 48
#endif
__attribute__((used, retain)) volatile const unsigned long __pad[LUA_PROBE_PAD_WORDS] = {1};

static const char CHUNK[] = "local t={} for i=1,20 do t[i]=i*i end return t[20]";

int capstone_main(void) {
  lua_State *L;
  int rc;
  lua_Integer result;

  if (__pad[0] != 1)
    return 1;

  SAY("LUA S1: entered\n");

  L = luaL_newstate();
  if (!L) {
    SAY("LUA FAIL: newstate returned NULL\n");
    return 2;
  }
  SAY("LUA S2: newstate ok\n");

  luaopen_base(L);
  SAY("LUA S3: base library opened\n");

  rc = luaL_loadbufferx(L, CHUNK, sizeof(CHUNK) - 1, "probe", 0);
  if (rc != LUA_OK) {
    SAY("LUA FAIL: load\n");
    return 3;
  }
  SAY("LUA S4: chunk compiled\n");

  rc = lua_pcall(L, 0, 1, 0);
  if (rc != LUA_OK) {
    SAY("LUA FAIL: pcall\n");
    return 4;
  }
  SAY("LUA S5: pcall ok\n");

  result = lua_tointegerx(L, -1, 0);
  if (result != 400) {
    SAY("LUA FAIL: wrong result\n");
    return 5;
  }

  /* The marker states the value, because "PASSED" on its own would also be
     printed by a build that returned the right thing for the wrong reason. */
  SAY("LUA OK: t[20] == 400, real interpreter on musl\n");
  lua_close(L);
  SAY("__CAPSTONE_LUA_PROBE_PASSED__\n");
  return 0;
}
