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
/* Each snippet gets a FRESH lua_State (newstate + base + close), so there is no
 * cross-chunk lua_settop plumbing to trip a separate fault -- the marker gap is
 * purely about the snippet. Markers 310 + k*10 + phase {0 newstate,1 load,2 pcall-pre,
 * 3 pcall-post}. A missing pcall-post (…2 with no …3) means snippet k faulted in
 * execution. */
static int ladder_one(const char *src, unsigned k) {
  DBG(310 + k * 10 + 0);
  lua_State *L = luaL_newstate();
  if (!L) {
    output_text("C");
    output_int(k);
    output_text(" newstate=NULL\n");
    return -1;
  }
  /* base enabled per snippet so print/assert/etc. resolve */
  luaL_requiref(L, LUA_GNAME, luaopen_base, 1);
  lua_pop(L, 1);
  const char *s = (const char *)__builtin_capstone_cap_delin((void *)src);
  unsigned long len = 0;
  while (s[len])
    len++;
  int st = luaL_loadbufferx(L, src, len, "=c", NULL);
  DBG(310 + k * 10 + 1);
  if (st != LUA_OK) {
    output_text("C");
    output_int(k);
    output_text(" load=");
    output_int(st);
    output_text("\n");
    lua_close(L);
    return st;
  }
  DBG(310 + k * 10 + 2); /* pcall-pre: the suspect boundary */
  st = lua_pcall(L, 0, 1, 0);
  DBG(310 + k * 10 + 3); /* pcall-post: MISSING for snippet k => it faulted */
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
  lua_close(L);
  return st;
}

/* Base-call bisection: control, global LOOKUP, CALL-no-arg, CALL-with-arg, a
 * different base fn. First snippet whose pcall-post marker is missing localises the
 * fault (lookup vs call vs arg handling vs a specific fn). */
static void run_lua_ladder(void) {
  DBG(300);
  ladder_one("return 1", 0);                    /* control: no base */
  ladder_one("local p=print return 1", 1);      /* GETTABUP _ENV.print, no call */
  ladder_one("print() return 1", 2);            /* CALL print, no args */
  ladder_one("print('x') return 1", 3);         /* CALL print, string arg (output) */
  ladder_one("assert(true) return 5", 4);       /* different base fn (assert) */
  ladder_one("return type(1)==nil and 0 or 6", 5); /* type() call, compare */
  DBG(399);
  output_text("LADDER done\n");
}
#endif /* LUA_CHUNK_LADDER */

/* luaossl #124 reproduced through REAL Lua (userdata + __gc + GC), not a pure-C
 * shim. Built with -DLUA_CDP_X509. This is the fidelity upgrade the interpreter
 * bring-up unlocks: the cross-domain double-free is driven by Lua's own garbage
 * collector calling finalizers, and revoke-on-free must catch the second owner's
 * stale refcount access. The C X509_STORE / SSL_CTX are minimal stubs (only the
 * memory lifecycle the bug depends on -- exactly what the pure-C shim already
 * distilled); everything cross-domain (the userdata handles, the __gc metamethods,
 * the GC-driven free) is now the genuine article.
 *
 *   store userdata __gc = xs__gc -> X509_STORE_free: rc 1->0 -> free -> REVOKE
 *   ctx   userdata __gc = sx__gc -> SSL_CTX_free -> X509_STORE_free(cert_store):
 *         reads/writes the refcount of the ALREADY-FREED store through the
 *         set0-aliased (now revoked) capability -> FAULT on Capstone (caught).
 * Control (-DLUA_CDP_NO_REVOKE): free does not revoke -> the second access
 * completes -> "CDP-MISS survived".
 *
 * Marker trace: 480 enter, 481 newstate, 482 registered, 483 loaded; 500/501
 * xs__gc (first free); 510 sx__gc pre-stale-access, 511 post (control only); 484
 * pcall returned (control only). With revoke, 510 is the last marker. */
#ifdef LUA_CDP_X509
extern void *malloc(unsigned long);
extern void free(void *);
extern void xlang_set_no_revoke(void);

#define X509_STORE_BYTES 152
#define REFCOUNT_OFF 136 /* the refcount word ASan names in the real bug */

typedef struct {
  unsigned char *cert_store; /* set0 alias to the X509_STORE (no up-ref) */
} MockSSL_CTX;

/* X509_STORE_free: decrement refcount, free at 0. The `--(*rc)` is the access
 * that faults when `store` has already been freed+revoked (the CDP contract
 * point). free() here is the REVOKING free. */
static void x509_store_free(unsigned char *store) {
  if (!store)
    return;
  volatile unsigned int *rc = (volatile unsigned int *)(store + REFCOUNT_OFF);
  if (--(*rc) == 0) /* stale READ+WRITE through a revoked cap -> FAULT */
    free(store);
}

static int xs__gc(lua_State *L) { /* store userdata __gc (FIRST free) */
  unsigned char **pp = (unsigned char **)lua_touserdata(L, 1);
  DBG(500);
  x509_store_free(*pp); /* rc 1->0 -> free -> REVOKE */
  DBG(501);
  return 0;
}

static int sx__gc(lua_State *L) { /* ctx userdata __gc (SSL_CTX_free, STALE) */
  MockSSL_CTX **pp = (MockSSL_CTX **)lua_touserdata(L, 1);
  MockSSL_CTX *ctx = *pp;
  DBG(510);
  x509_store_free(ctx->cert_store); /* cert_store revoked -> stale rc access -> FAULT */
  DBG(511);
  free(ctx);
  return 0;
}

static int store_new(lua_State *L) {
  unsigned char *store = (unsigned char *)malloc(X509_STORE_BYTES);
  for (int i = 0; i < X509_STORE_BYTES; i++)
    store[i] = 0;
  *(volatile unsigned int *)(store + REFCOUNT_OFF) = 1; /* refcount = 1, no up-ref */
  unsigned char **ud =
      (unsigned char **)lua_newuserdatauv(L, sizeof(unsigned char *), 0);
  *ud = store;
  luaL_setmetatable(L, "x509.store");
  return 1;
}

static int ctx_new(lua_State *L) {
  MockSSL_CTX *ctx = (MockSSL_CTX *)malloc(sizeof(MockSSL_CTX));
  ctx->cert_store = 0;
  MockSSL_CTX **ud =
      (MockSSL_CTX **)lua_newuserdatauv(L, sizeof(MockSSL_CTX *), 0);
  *ud = ctx;
  luaL_setmetatable(L, "ssl.context");
  return 1;
}

/* ctx:setStore(store) -- the vulnerable set0: co-own the store, NO refcount
 * up-ref, and keep NO Lua reference to the store userdata (so the GC frees it
 * independently while the ctx still holds the raw C pointer). */
static int ctx_setStore(lua_State *L) {
  MockSSL_CTX **cud = (MockSSL_CTX **)lua_touserdata(L, 1);
  unsigned char **sud = (unsigned char **)lua_touserdata(L, 2);
  (*cud)->cert_store = *sud; /* alias; refcount STILL 1 */
  return 0;
}

/* Drive GC from C so the reproduction needs no base library. */
static int docollect(lua_State *L) {
  lua_gc(L, LUA_GCCOLLECT, 0);
  lua_gc(L, LUA_GCCOLLECT, 0); /* second pass: ensure queued finalizers run */
  return 0;
}

static const char CDP_CHUNK[] =
    "local ctx = ctx_new()\n"
    "do local store = store_new() setStore(ctx, store) end\n"
    "docollect()\n"  /* FIRST free: store __gc (xs__gc) -> free -> REVOKE */
    "ctx = nil\n"
    "docollect()\n"  /* SECOND: ctx __gc (sx__gc) -> stale refcount access -> FAULT */
    "return 1\n";    /* reached only if the double-free SURVIVED (control) */

static void reg_gc_meta(lua_State *L, const char *name, lua_CFunction gc) {
  luaL_newmetatable(L, name);
  lua_pushcfunction(L, gc);
  lua_setfield(L, -2, "__gc");
  lua_pop(L, 1);
}

static void run_lua_cdp_x509(void) {
#ifdef LUA_CDP_NO_REVOKE
  xlang_set_no_revoke(); /* control: free() does not revoke */
  output_text("CDP mode: NO-REVOKE (control)\n");
#else
  output_text("CDP mode: REVOKE\n");
#endif
  DBG(480);
  lua_State *L = luaL_newstate();
  if (!L) {
    output_text("CDP newstate=NULL\n");
    return;
  }
  DBG(481);
  reg_gc_meta(L, "x509.store", xs__gc);
  reg_gc_meta(L, "ssl.context", sx__gc);
  lua_pushcfunction(L, store_new);
  lua_setglobal(L, "store_new");
  lua_pushcfunction(L, ctx_new);
  lua_setglobal(L, "ctx_new");
  lua_pushcfunction(L, ctx_setStore);
  lua_setglobal(L, "setStore");
  lua_pushcfunction(L, docollect);
  lua_setglobal(L, "docollect");
  DBG(482);
  int st = luaL_loadbufferx(L, CDP_CHUNK, sizeof(CDP_CHUNK) - 1, "=cdp", NULL);
  DBG(483);
  if (st != LUA_OK) {
    output_text("CDP load fail: ");
    output_text(lua_tostring(L, -1));
    output_text("\n");
    return;
  }
  st = lua_pcall(L, 0, 1, 0);
  DBG(484); /* only reached if the stale access did NOT fault */
  if (st == LUA_OK)
    output_text("CDP-MISS: double-free survived (no fault)\n");
  else {
    output_text("CDP runtime error (unexpected): ");
    output_text(lua_tostring(L, -1));
    output_text("\n");
  }
  lua_close(L);
}
#endif /* LUA_CDP_X509 */

/* The 11 single-object cross-domain UAFs of the corpus, uniform shape (only
 * openssl_ctx and luaossl-124 are double-frees; luaossl-124 is LUA_CDP_X509 above).
 * Each: a C object owned by a Lua-GC userdata (its __gc frees it) with a BORROWED
 * stale pointer cached in a second handle that keeps NO reference to the owner. The
 * GC frees+REVOKES the object; the later stale deref (read/write a field at OFF)
 * faults. (size,off,write) are the real bug's ASan values, kept for fidelity; the
 * catch is offset-independent (revoke invalidates the whole block).
 *
 * Built with -DLUA_CDP_UAF. Revoke (default): runs ONE case (-DCDP_UAF_CASE=k) ->
 * faults=CAUGHT. Control (+LUA_CDP_NO_REVOKE): runs ALL cases fresh-state -> each
 * completes=MISS. Markers: 500/501 owner __gc (free/revoke), 510/511 stale deref
 * pre/post, 520 pcall returned (survived). */
#ifdef LUA_CDP_UAF
extern void *malloc(unsigned long);
extern void free(void *);
extern void xlang_set_no_revoke(void);

struct uaf_case {
  const char *name;
  unsigned long size, off;
  int write;
};
static const struct uaf_case UAF_CASES[] = {
    {"curl_multi_backptr", 96, 64, 1}, /* lua-curl-80  : ->L back-pointer WRITE */
    {"ffi_closure", 56, 32, 0},        /* cffi-lua-57  : ->fref READ */
    {"ldbus_message", 64, 0, 0},       /* ldbus-20     : arg-type tag READ */
    {"lgi_cairo_region", 32, 4, 0},    /* lgi-122      : extents READ */
    {"lgi_garray", 16, 0, 0},          /* lgi-65       : ->len READ */
    {"lmdb_value", 8192, 0, 0},        /* lmdb         : borrowed v->data READ */
    {"luv_costate", 208, 24, 0},       /* luv-503      : freed lua_State field READ */
    {"pgconn", 1056, 376, 0},          /* luadbi-35    : PQstatus READ */
    {"sdl_window", 240, 0, 0},         /* lua-sdl2-75  : window->magic READ */
    {"tvbuff", 72, 16, 0},             /* wireshark    : tvbuff field READ */
    {"uv_fs", 64, 0, 0},               /* luv-696      : freed request READ */
};
#define N_UAF_CASES (sizeof(UAF_CASES) / sizeof(UAF_CASES[0]))
#ifndef CDP_UAF_CASE
#define CDP_UAF_CASE 0
#endif
static unsigned g_uaf_idx;

static int uaf_owner_gc(lua_State *L) { /* owner __gc -> free -> REVOKE */
  void **pp = (void **)lua_touserdata(L, 1);
  DBG(500);
  free(*pp);
  DBG(501);
  return 0;
}
static int uaf_new(lua_State *L) {
  unsigned long sz = UAF_CASES[g_uaf_idx].size;
  unsigned char *obj = (unsigned char *)malloc(sz);
  for (unsigned long i = 0; i < sz; i++)
    obj[i] = 0;
  void **ud = (void **)lua_newuserdatauv(L, sizeof(void *), 0);
  *ud = obj;
  luaL_setmetatable(L, "uaf.owner");
  return 1;
}
static int uaf_borrow(lua_State *L) { /* view: cache raw ptr, NO owner ref, NO __gc */
  void **owner = (void **)lua_touserdata(L, 1);
  void **ud = (void **)lua_newuserdatauv(L, sizeof(void *), 0);
  *ud = *owner;
  return 1;
}
static int uaf_use(lua_State *L) { /* stale deref of the revoked ptr */
  void **view = (void **)lua_touserdata(L, 1);
  DBG(510); /* before the offset computation, which is where a cincoffset case faults */
  unsigned char *p = (unsigned char *)(*view) + UAF_CASES[g_uaf_idx].off;
  if (UAF_CASES[g_uaf_idx].write)
    *(volatile unsigned int *)p = 0xA5A5A5A5u; /* stale WRITE */
  else {
    volatile unsigned int s = *(volatile unsigned int *)p; /* stale READ */
    (void)s;
  }
  DBG(511);
  return 0;
}
static int uaf_docollect(lua_State *L) {
  lua_gc(L, LUA_GCCOLLECT, 0);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 0;
}

static const char UAF_CHUNK[] =
    "local h = uaf_new()\n"
    "local v = uaf_borrow(h)\n" /* v caches h's raw C pointer, no ref to h */
    "h = nil\n"
    "docollect()\n"  /* h __gc -> free -> REVOKE (invalidates v's cached ptr) */
    "uaf_use(v)\n"   /* stale deref -> FAULT (revoke) / completes (control) */
    "return 1\n";

static void uaf_one(unsigned idx) {
  g_uaf_idx = idx;
  output_text("UAF[");
  output_int((long long)idx);
  output_text("] ");
  output_text(UAF_CASES[idx].name);
  output_text(": ");
  lua_State *L = luaL_newstate();
  if (!L) {
    output_text("newstate=NULL (arena?)\n");
    return;
  }
  luaL_newmetatable(L, "uaf.owner");
  lua_pushcfunction(L, uaf_owner_gc);
  lua_setfield(L, -2, "__gc");
  lua_pop(L, 1);
  lua_pushcfunction(L, uaf_new);
  lua_setglobal(L, "uaf_new");
  lua_pushcfunction(L, uaf_borrow);
  lua_setglobal(L, "uaf_borrow");
  lua_pushcfunction(L, uaf_use);
  lua_setglobal(L, "uaf_use");
  lua_pushcfunction(L, uaf_docollect);
  lua_setglobal(L, "docollect");
  int st = luaL_loadbufferx(L, UAF_CHUNK, sizeof(UAF_CHUNK) - 1, "=uaf", NULL);
  if (st != LUA_OK) {
    output_text("load fail\n");
    lua_close(L);
    return;
  }
  st = lua_pcall(L, 0, 1, 0);
  DBG(520); /* reached only if the stale deref did NOT fault */
  if (st == LUA_OK)
    output_text("MISS survived\n");
  else {
    output_text("ERR: ");
    output_text(lua_tostring(L, -1));
    output_text("\n");
  }
  lua_close(L);
}

static void run_lua_cdp_uaf(void) {
#ifdef LUA_CDP_NO_REVOKE
  xlang_set_no_revoke();
  output_text("UAF mode: NO-REVOKE (control), all cases\n");
  for (unsigned i = 0; i < N_UAF_CASES; i++)
    uaf_one(i);
  output_text("UAF-LADDER done\n");
#else
  output_text("UAF mode: REVOKE, one case\n");
  uaf_one((unsigned)CDP_UAF_CASE);
#endif
}
#endif /* LUA_CDP_UAF */

/* lua-openssl #141: an EVP_CIPHER_CTX co-owned by a cipher userdata's close() and
 * its __gc. close() frees the ctx WITHOUT nulling it (the bug); __gc then frees it
 * again, reading the freed ctx first. Single-object double-free (cf. luaossl-124
 * which is two userdata). Built with -DLUA_CDP_OPENSSL; +LUA_CDP_NO_REVOKE control.
 * Markers: 500/501 close (free/revoke), 510/511 __gc stale read pre/post, 484 done. */
#ifdef LUA_CDP_OPENSSL
extern void *malloc(unsigned long);
extern void free(void *);
extern void xlang_set_no_revoke(void);
#define OSSL_CTX_BYTES 168

static int ossl_close(lua_State *L) { /* c:close() -> free, NOT nulled (the bug) */
  void **pp = (void **)lua_touserdata(L, 1);
  if (*pp) {
    DBG(500);
    free(*pp); /* crossing 1: free -> REVOKE */
    DBG(501);
  }
  return 0;
}
static int ossl_gc(lua_State *L) { /* __gc -> free AGAIN, reading the freed ctx */
  void **pp = (void **)lua_touserdata(L, 1);
  unsigned char *ctx = (unsigned char *)*pp;
  if (ctx) {
    DBG(510);
    volatile unsigned int s = *(volatile unsigned int *)ctx; /* crossing 2: STALE read */
    (void)s;
    DBG(511);
    free(ctx);
  }
  return 0;
}
static int ossl_new(lua_State *L) {
  unsigned char *ctx = (unsigned char *)malloc(OSSL_CTX_BYTES);
  for (int i = 0; i < OSSL_CTX_BYTES; i++)
    ctx[i] = 0;
  void **ud = (void **)lua_newuserdatauv(L, sizeof(void *), 0);
  *ud = ctx;
  luaL_setmetatable(L, "ossl.cipher");
  return 1;
}
static int ossl_docollect(lua_State *L) {
  lua_gc(L, LUA_GCCOLLECT, 0);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 0;
}
static const char OSSL_CHUNK[] =
    "local c = ossl_new()\n"
    "close(c)\n"    /* crossing 1: free ctx -> REVOKE */
    "c = nil\n"
    "docollect()\n" /* crossing 2: __gc reads the freed ctx -> FAULT */
    "return 1\n";

static void run_lua_cdp_openssl(void) {
#ifdef LUA_CDP_NO_REVOKE
  xlang_set_no_revoke();
  output_text("OSSL mode: NO-REVOKE (control)\n");
#else
  output_text("OSSL mode: REVOKE\n");
#endif
  DBG(480);
  lua_State *L = luaL_newstate();
  if (!L) {
    output_text("OSSL newstate=NULL\n");
    return;
  }
  DBG(481);
  luaL_newmetatable(L, "ossl.cipher");
  lua_pushcfunction(L, ossl_gc);
  lua_setfield(L, -2, "__gc");
  lua_pop(L, 1);
  lua_pushcfunction(L, ossl_new);
  lua_setglobal(L, "ossl_new");
  lua_pushcfunction(L, ossl_close);
  lua_setglobal(L, "close");
  lua_pushcfunction(L, ossl_docollect);
  lua_setglobal(L, "docollect");
  DBG(482);
  int st = luaL_loadbufferx(L, OSSL_CHUNK, sizeof(OSSL_CHUNK) - 1, "=ossl", NULL);
  DBG(483);
  if (st != LUA_OK) {
    output_text("OSSL load fail\n");
    return;
  }
  st = lua_pcall(L, 0, 1, 0);
  DBG(484); /* reached only if the stale read did NOT fault */
  if (st == LUA_OK)
    output_text("OSSL-MISS: double-free survived\n");
  else {
    output_text("OSSL ERR: ");
    output_text(lua_tostring(L, -1));
    output_text("\n");
  }
  lua_close(L);
}
#endif /* LUA_CDP_OPENSSL */

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

#ifdef LUA_CDP_UAF
  run_lua_cdp_uaf();
#elif defined(LUA_CDP_X509)
  run_lua_cdp_x509();
#elif defined(LUA_CHUNK_LADDER)
  run_lua_ladder();
#elif defined(LUA_STAGE)
  run_lua_staged();
#else
  run_lua();
#endif
}
