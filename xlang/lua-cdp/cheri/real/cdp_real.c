/* The 13 lua-cdp corpus cases reproduced through REAL Lua on CheriBSD purecap --
 * the CHERI half of the fair comparison, mirroring the Capstone LUA_CDP_* domains
 * (../../capstone-lua/lua_domain.c). One binary, dispatched on its own argv[0]
 * basename (so a single build is copied to the 13 row names), so the reproduction
 * logic is identical across cases and platforms.
 *
 * Each case is a cross-domain UAF/double-free driven by Lua's GC calling __gc on
 * real userdata. Under CheriBSD revocation (eager) the stale access through the
 * revoked capability faults (SIGPROT -> nonzero exit = CAUGHT); with revocation off
 * (spatial) or async (default) it completes and prints MISS. The wrapped C object is
 * a minimal stub (only the memory lifecycle the bug depends on); everything
 * cross-domain -- the userdata handles, the __gc metamethods, the GC-driven free --
 * is the genuine interpreter. (size,off,write) are each bug's real ASan values.
 */
#include "lua.h"
#include "lauxlib.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

enum { KIND_UAF, KIND_X509, KIND_OSSL };
struct C { const char *name; int kind; unsigned long size, off; int write; };
static const struct C CASES[] = {
    {"luac_curl_multi_backptr_uaf", KIND_UAF, 96, 64, 1},
    {"luac_ffi_closure_uaf", KIND_UAF, 56, 32, 0},
    {"luac_ldbus_message_uaf", KIND_UAF, 64, 0, 0},
    {"luac_lgi_cairo_region_uaf", KIND_UAF, 32, 4, 0},
    {"luac_lgi_garray_uaf", KIND_UAF, 16, 0, 0},
    {"luac_lmdb_value_uaf", KIND_UAF, 8192, 0, 0},
    {"luac_luv_costate_uar", KIND_UAF, 208, 24, 0},
    {"luac_pgconn_uaf", KIND_UAF, 1056, 376, 0},
    {"luac_sdl_window_uaf", KIND_UAF, 240, 0, 0},
    {"luac_tvbuff_uaf", KIND_UAF, 72, 16, 0},
    {"luac_uv_fs_uaf", KIND_UAF, 64, 0, 0},
    {"luac_x509_store_dblfree", KIND_X509, 152, 136, 0},
    {"luac_openssl_ctx_uaf", KIND_OSSL, 168, 0, 0},
};
#define NCASES ((int)(sizeof(CASES) / sizeof(CASES[0])))
static const struct C *C_; /* the selected case */

/* drive GC from C so no base library is needed */
static int docollect(lua_State *L) {
  lua_gc(L, LUA_GCCOLLECT, 0);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 0;
}

/* ---- generic single-object borrowed-view UAF ---- */
static int uaf_owner_gc(lua_State *L) {
  void **pp = (void **)lua_touserdata(L, 1);
  free(*pp); /* -> REVOKE under eager */
  return 0;
}
static int uaf_new(lua_State *L) {
  unsigned char *o = (unsigned char *)malloc(C_->size);
  memset(o, 0, C_->size);
  void **ud = (void **)lua_newuserdatauv(L, sizeof(void *), 0);
  *ud = o;
  luaL_setmetatable(L, "uaf.owner");
  return 1;
}
static int uaf_borrow(lua_State *L) {
  void **owner = (void **)lua_touserdata(L, 1);
  void **ud = (void **)lua_newuserdatauv(L, sizeof(void *), 0);
  *ud = *owner; /* cache raw ptr, no owner ref, no __gc */
  return 1;
}
static int uaf_use(lua_State *L) {
  void **view = (void **)lua_touserdata(L, 1);
  unsigned char *p = (unsigned char *)(*view) + C_->off;
  if (C_->write)
    *(volatile unsigned int *)p = 0xA5A5A5A5u; /* stale WRITE -> FAULT if revoked */
  else {
    volatile unsigned int s = *(volatile unsigned int *)p; /* stale READ */
    (void)s;
  }
  return 0;
}

/* ---- luaossl-124: two userdata co-own one X509_STORE (set0 double-free) ---- */
typedef struct { unsigned char *cert_store; } MockSSL_CTX;
static void x509_store_free(unsigned char *s) {
  if (!s) return;
  volatile unsigned int *rc = (volatile unsigned int *)(s + C_->off);
  if (--(*rc) == 0) free(s);
}
static int xs__gc(lua_State *L) { unsigned char **pp = (unsigned char **)lua_touserdata(L, 1); x509_store_free(*pp); return 0; }
static int sx__gc(lua_State *L) { MockSSL_CTX **pp = (MockSSL_CTX **)lua_touserdata(L, 1); MockSSL_CTX *c = *pp; x509_store_free(c->cert_store); free(c); return 0; }
static int store_new(lua_State *L) {
  unsigned char *s = (unsigned char *)malloc(C_->size);
  memset(s, 0, C_->size);
  *(volatile unsigned int *)(s + C_->off) = 1;
  unsigned char **ud = (unsigned char **)lua_newuserdatauv(L, sizeof(void *), 0);
  *ud = s;
  luaL_setmetatable(L, "x509.store");
  return 1;
}
static int ctx_new(lua_State *L) {
  MockSSL_CTX *c = (MockSSL_CTX *)malloc(sizeof(MockSSL_CTX));
  c->cert_store = 0;
  MockSSL_CTX **ud = (MockSSL_CTX **)lua_newuserdatauv(L, sizeof(void *), 0);
  *ud = c;
  luaL_setmetatable(L, "ssl.context");
  return 1;
}
static int setStore(lua_State *L) { MockSSL_CTX **c = (MockSSL_CTX **)lua_touserdata(L, 1); unsigned char **s = (unsigned char **)lua_touserdata(L, 2); (*c)->cert_store = *s; return 0; }

/* ---- lua-openssl-141: close() frees the ctx (not nulled); __gc reads+frees it ---- */
static int ossl_close(lua_State *L) { void **pp = (void **)lua_touserdata(L, 1); if (*pp) free(*pp); return 0; }
static int ossl_gc(lua_State *L) {
  void **pp = (void **)lua_touserdata(L, 1);
  unsigned char *ctx = (unsigned char *)*pp;
  /* The CDP contract point is the STALE READ of the freed ctx (the ASan HUF). We do
   * NOT issue the actual second free: on a real (CheriBSD) allocator an unconditional
   * double-free trips the allocator's own double-free detection (SIGABRT) under async,
   * contaminating the revocation measurement. Reading the revoked capability is what
   * revocation must police -- faults under eager (CAUGHT), completes under async/off. */
  if (ctx) { volatile unsigned int s = *(volatile unsigned int *)ctx; (void)s; }
  return 0;
}
static int ossl_new(lua_State *L) {
  unsigned char *c = (unsigned char *)malloc(C_->size);
  memset(c, 0, C_->size);
  void **ud = (void **)lua_newuserdatauv(L, sizeof(void *), 0);
  *ud = c;
  luaL_setmetatable(L, "ossl.cipher");
  return 1;
}

static void reg(lua_State *L, const char *fn, lua_CFunction f) { lua_pushcfunction(L, f); lua_setglobal(L, fn); }
static void meta(lua_State *L, const char *n, lua_CFunction gc) { luaL_newmetatable(L, n); lua_pushcfunction(L, gc); lua_setfield(L, -2, "__gc"); lua_pop(L, 1); }

int main(int argc, char **argv) {
  const char *base = argv[0] ? argv[0] : "";
  const char *slash = strrchr(base, '/');
  if (slash) base = slash + 1;
  C_ = NULL;
  for (int i = 0; i < NCASES; i++)
    if (strcmp(CASES[i].name, base) == 0) { C_ = &CASES[i]; break; }
  if (!C_) { fprintf(stderr, "unknown case '%s'\n", base); return 3; }

  lua_State *L = luaL_newstate();
  reg(L, "docollect", docollect);
  const char *chunk;
  if (C_->kind == KIND_UAF) {
    meta(L, "uaf.owner", uaf_owner_gc);
    reg(L, "uaf_new", uaf_new); reg(L, "uaf_borrow", uaf_borrow); reg(L, "uaf_use", uaf_use);
    chunk = "local h=uaf_new() local v=uaf_borrow(h) h=nil docollect() uaf_use(v) return 1";
  } else if (C_->kind == KIND_X509) {
    meta(L, "x509.store", xs__gc); meta(L, "ssl.context", sx__gc);
    reg(L, "store_new", store_new); reg(L, "ctx_new", ctx_new); reg(L, "setStore", setStore);
    chunk = "local ctx=ctx_new() do local s=store_new() setStore(ctx,s) end docollect() ctx=nil docollect() return 1";
  } else { /* KIND_OSSL */
    meta(L, "ossl.cipher", ossl_gc);
    reg(L, "ossl_new", ossl_new); reg(L, "ossl_close", ossl_close);
    chunk = "local c=ossl_new() ossl_close(c) c=nil docollect() return 1";
  }
  if (luaL_dostring(L, chunk) != LUA_OK) { printf("%s CDP-ERR: %s\n", base, lua_tostring(L, -1)); return 2; }
  printf("%s CDP-MISS survived\n", base); /* reached only if the stale access did NOT fault */
  lua_close(L);
  return 0;
}
