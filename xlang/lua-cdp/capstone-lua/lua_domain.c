/* A minimal REAL-Lua-5.4.7 Capstone freestanding domain.
 *
 * Two halves, like xlang_shim_domain.c: the host (lua_host.c) shares three regions
 * in order — [0] hostcall metadata, [1] text payload, [2] the LINEAR arena grant
 * that becomes the revoking allocator's arena — then call_dom() runs this.
 *
 * Unlike the xlang shim column (hand-written C stand-ins), this links the ACTUAL
 * reference Lua interpreter: luaL_newstate() with Lua's default realloc/free
 * allocator, wired to the tag-preserving revoking realloc in revoke_arena_domain.c.
 * It loads a trivial chunk from memory (luaL_loadbufferx, NOT the file loader),
 * runs it under lua_pcall, and writes the integer result into the payload so the
 * host prints it. The chunk builds a 20-element table (forcing the array part to be
 * realloc'd and MOVED) — the exact event that would strip a capability tag if the
 * realloc copied bytes instead of capability words.
 *
 * -O0 like every domain here (start.S / the allocator contract).
 */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

/* The hostcall metadata layout, shared with the host. {phase,opcode,offset,length,
 * result,error}; the host flushes `length` bytes of the payload region. */
#include "sqlite_hostcall.h"

#define CAPSTONE_DPI_REGION_SHARE 1U
#define XLANG_HC_REGION_SIZE SQLITE_HC_REGION_SIZE

extern void xlang_arena_init(void *grant); /* revoke_arena_domain.c owns the arena */

/* QEMU-only console markers that SURVIVE a wedge. Payload markers are read by the host
 * only AFTER call_dom() returns, so a hang loses them; csdebugprint (funct7 0x43) writes
 * straight to the emulator console, so stage N shows as "...cursor=N..." even if the very
 * next call never returns. Built only with -DLUA_DBG_STAGE; never for the board. */
#ifdef LUA_DBG_STAGE
#define DBG(n) __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0" ::"r"((unsigned long)(n)))
#else
#define DBG(n) ((void)0)
#endif

/* Skip luaL_openlibs(base) at RUNTIME (not compile time) with -DLUA_SKIP_BASE. Keeping
 * the luaopen_base call in the code -- behind this volatile flag so the optimiser cannot
 * fold the branch away -- keeps lbaselib LINKED under --gc-sections, so the domain image
 * stays byte-layout-compatible with the base-enabled build (same globals, same
 * descriptor). The demo chunk uses only core (locals, a table, a numeric for, arithmetic),
 * so base is optional; skipping it avoids the luaopen_base wedge while proving the core
 * interpreter runs a chunk end to end. */
static volatile int lua_skip_base =
#ifdef LUA_SKIP_BASE
    1;
#else
    0;
#endif

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

/* Append n bytes to the payload region. Both the source and the payload are
 * delin'd first (matches xlang_shim_domain.c): the source may be a Lua string in
 * the LINEAR arena, and the payload capability is shared LIN. */
static void payload_put(const char *src0, unsigned long n) {
  if (!hostcall_metadata || !hostcall_payload)
    return;
  const char *src = (const char *)__builtin_capstone_cap_delin((void *)src0);
  char *payload = (char *)__builtin_capstone_cap_delin((void *)hostcall_payload);
  unsigned long off = hostcall_metadata->length;
  for (unsigned long i = 0; i < n && off + 1 < XLANG_HC_REGION_SIZE; i++)
    payload[off++] = src[i];
  hostcall_metadata->length = off;
}

static void output_text(const char *s) {
  unsigned long n = 0;
  const char *p = (const char *)__builtin_capstone_cap_delin((void *)s);
  while (p[n])
    n++;
  payload_put(s, n);
}

/* Decimal print of a signed 64-bit value straight into the payload — deliberately
 * independent of snprintf, so the load-bearing RESULT line cannot be mis-shown by a
 * formatter bug. Lua computes the value; this just renders it. */
static void output_int(long long v) {
  char buf[24];
  int i = 0;
  unsigned long long m = v < 0 ? (unsigned long long)(-(v + 1)) + 1ULL
                               : (unsigned long long)v;
  char rev[24];
  int rn = 0;
  if (m == 0)
    rev[rn++] = '0';
  while (m) {
    rev[rn++] = (char)('0' + m % 10);
    m /= 10;
  }
  if (v < 0)
    buf[i++] = '-';
  while (rn)
    buf[i++] = rev[--rn];
  payload_put(buf, (unsigned long)i);
}

/* print()/io.write route through fwrite(stdout) -> this sink (see lua_libc.c). */
void lua_host_write(const char *s, unsigned long n) { payload_put(s, n); }

/* Lua calls abort() only from luaD_throw with no protected frame (a panic). We
 * always run under lua_pcall, so errorJmp is set and this is unreachable — but it
 * must be defined to link. Freestanding: no exit(); mark the payload and spin. */
void abort(void) {
  output_text("LUA-FAIL abort() (unprotected Lua panic)\n");
  for (;;)
    ;
}

/* The chunk: allocate a table and fill 20 integer slots, forcing the array part to
 * be realloc'd and MOVED several times, then return t[20]. 20*20 == 400. */
static const char LUA_CHUNK[] =
    "local t={} for i=1,20 do t[i]=i*i end return t[20]";

static void run_lua(void) {
  lua_State *L = luaL_newstate();
  if (!L) {
    output_text("LUA-FAIL newstate returned NULL (arena depleted?)\n");
    return;
  }
#ifndef LUA_SKIP_BASE
  /* Just the base library — luaL_openlibs is heavy and unneeded here.
   * -DLUA_SKIP_BASE runs a pure-core chunk without opening base (the base lib's
   * static base_funcs[] cap-globals are the current cap-init blocker). */
  luaL_requiref(L, LUA_GNAME, luaopen_base, 1);
  lua_pop(L, 1); /* drop the module table luaL_requiref leaves on the stack */
#endif

  int st = luaL_loadbufferx(L, LUA_CHUNK, sizeof(LUA_CHUNK) - 1, "=chunk", NULL);
  if (st != LUA_OK) {
    output_text("LUA-FAIL load error: ");
    output_text(lua_tostring(L, -1));
    output_text("\n");
    lua_close(L);
    return;
  }
  st = lua_pcall(L, 0, 1, 0);
  if (st != LUA_OK) {
    output_text("LUA-FAIL runtime error: ");
    output_text(lua_tostring(L, -1));
    output_text("\n");
    lua_close(L);
    return;
  }
  int isint = 0;
  long long r = lua_tointegerx(L, -1, &isint);
  output_text("LUA-OK result=");
  if (isint)
    output_int(r);
  else
    output_text("(non-integer)");
  output_text(" expected=400\n");
  lua_close(L);
}

/* Bisection harness (CLAUDE.md batch-variants). Built only with -DLUA_STAGE=k, in
 * a SEPARATE function so the real run_lua() above stays byte-identical. Each stage
 * does one more step, writes a marker, and RETURNS — so a build always yields data
 * even when the next step hangs. Run stages ascending in one boot: the last marker
 * flushed is the last step that returned; the next step is the wedge. */
#ifdef LUA_STAGE
static void run_lua_staged(void) {
  DBG(900);
  output_text("S0 enter\n");
#if LUA_STAGE >= 1
  DBG(910);
  lua_State *L = luaL_newstate();
  DBG(911);
  if (!L) {
    output_text("S1 newstate=NULL\n");
    return;
  }
  output_text("S1 newstate ok\n");
#endif
#if LUA_STAGE >= 2
  if (!lua_skip_base) {
    DBG(920);
    luaL_requiref(L, LUA_GNAME, luaopen_base, 1);
    lua_pop(L, 1);
    DBG(921);
    output_text("S2 base ok\n");
  } else {
    output_text("S2 base SKIPPED\n");
  }
#endif
#if LUA_STAGE >= 3
  DBG(930);
  int st = luaL_loadbufferx(L, LUA_CHUNK, sizeof(LUA_CHUNK) - 1, "=chunk", NULL);
  DBG(931);
  output_text("S3 load rc=");
  output_int(st);
  output_text("\n");
  if (st != LUA_OK) {
    output_text(lua_tostring(L, -1));
    output_text("\n");
    return;
  }
#endif
#if LUA_STAGE >= 4
  DBG(940);
  st = lua_pcall(L, 0, 1, 0);
  DBG(941);
  output_text("S4 pcall rc=");
  output_int(st);
  output_text("\n");
  if (st != LUA_OK) {
    output_text(lua_tostring(L, -1));
    output_text("\n");
    return;
  }
#endif
#if LUA_STAGE >= 5
  DBG(950);
  int isint = 0;
  long long r = lua_tointegerx(L, -1, &isint);
  DBG(951);
  output_text("LUA-OK result=");
  output_int(r);
  output_text(" expected=400\n");
  DBG(952);
#endif
}
#endif /* LUA_STAGE */

/* Chunk ladder (CLAUDE.md batch-variants, applied to the INPUT instead of to a
 * single chunk's stages). Built with -DLUA_CHUNK_LADDER. Runs a spectrum of chunks
 * from trivial to the full demo in ONE lua_State, each with a csdebugprint marker
 * before/after its pcall, so a HANG localises to an opcode class: the first chunk
 * whose pcall-post marker is missing is the wedge. Ordered cheap -> complex so the
 * first non-returning chunk is the bisection point. Separate function, so run_lua /
 * run_lua_staged stay byte-identical. Marker scheme: 310 + k*10 + phase, phase in
 * {0 load-pre, 1 load-post, 2 pcall-pre, 3 pcall-post}; 300/301 newstate, 399 done.
 * If even chunk 0 ("return 1") hangs -> the call machinery (luaD_precall/luaV_execute
 * entry), not any opcode. */
#ifdef LUA_CHUNK_LADDER
static int ladder_one(lua_State *L, const char *src, unsigned k) {
  DBG(310 + k * 10 + 0);
  const char *s = (const char *)__builtin_capstone_cap_delin((void *)src);
  unsigned long len = 0;
  while (s[len])
    len++;
  int st = luaL_loadbufferx(L, src, len, "=c", NULL);
  DBG(310 + k * 10 + 1);
  output_text("C");
  output_int(k);
  output_text(" load=");
  output_int(st);
  output_text("\n");
  if (st != LUA_OK) {
    lua_settop(L, 0);
    return st;
  }
  DBG(310 + k * 10 + 2); /* pcall-pre: the suspect boundary */
  st = lua_pcall(L, 0, 1, 0);
  DBG(310 + k * 10 + 3); /* pcall-post: MISSING for chunk k => chunk k hung */
  output_text("C");
  output_int(k);
  output_text(" pcall=");
  output_int(st);
  if (st == LUA_OK) {
    int ii = 0;
    long long r = lua_tointegerx(L, -1, &ii);
    output_text(" r=");
    if (ii)
      output_int(r);
    else
      output_text("?");
  }
  output_text("\n");
  lua_settop(L, 0);
  return st;
}

static void run_lua_ladder(void) {
  DBG(300);
  lua_State *L = luaL_newstate();
  if (!L) {
    output_text("LADDER newstate=NULL\n");
    return;
  }
  DBG(301);
  ladder_one(L, "return 1", 0);                       /* RETURN1 only */
  ladder_one(L, "return 1+2", 1);                     /* integer ADD */
  ladder_one(L, "local x=5 return x", 2);             /* locals, MOVE */
  ladder_one(L, "return 3*4", 3);                     /* integer MUL */
  ladder_one(L, "local t={} return 0", 4);            /* NEWTABLE + GC */
  ladder_one(L, "local t={} t[1]=9 return t[1]", 5);  /* SETI / GETI */
  ladder_one(L, "local s=0 for i=1,3 do s=s+i end return s", 6); /* FORPREP/FORLOOP */
  ladder_one(L, "local t={} for i=1,20 do t[i]=i*i end return t[20]", 7); /* full */
  DBG(399);
  output_text("LADDER done\n");
}
#endif /* LUA_CHUNK_LADDER */

void domain_main(void *arg, unsigned func) {
  DBG(700 + func); /* every entry: 701 = share, 700 = run (survives a wedge) */
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (shared_region_count == 0)
      hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)arg;
    else if (shared_region_count == 1)
      hostcall_payload = (volatile char *)arg;
    else if (shared_region_count == 2)
      xlang_arena_init(arg); /* the LINEAR grant becomes the allocator's arena */
    ++shared_region_count;
    DBG(710 + shared_region_count); /* 711/712/713 after share 0/1/2 handled */
    return;
  }
  DBG(800); /* reached the run path (all shares handled) */

  if (hostcall_metadata)
    hostcall_metadata->length = 0;

#ifdef LUA_DBG_BASE
  /* QEMU-only csdebugprint (funct7 0x43): prints this image's code capability as
   * "Print = Cap(type, perms, cursor, base, end)" to the emulator console BEFORE
   * any fault, so a faulting run still reveals its runtime load base for mapping a
   * fault pc back to a static address. Never build this for the board. */
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0" ::"r"((void *)&domain_main));
#endif

#ifdef LUA_CHUNK_LADDER
  run_lua_ladder();
#elif defined(LUA_STAGE)
  run_lua_staged();
#else
  run_lua();
#endif
}
