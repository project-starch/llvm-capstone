/* minilmdb.c — a DELIBERATELY-UNSAFE minimal Lua binding to LMDB, written to
 * reproduce the documented-contract violation in lmdb.h:
 *
 *   "Values returned from the database are valid only until a subsequent
 *    update operation, or the end of the transaction. Do not modify or free
 *    them, they commonly point into the database itself."   (lmdb.h:249-251)
 *
 * A safe binding (e.g. shmul/lightningmdb) copies the value out with
 * lua_pushlstring the instant mdb_get returns, while the pointer is still
 * valid. This one does the opposite: `txn:get(k)` returns a DEFERRED value
 * handle that merely BORROWS the MDB_val {pointer,size} into the transaction's
 * page. Reading that handle (`val:read()`) after the transaction ends
 * dereferences a pointer into a page the transaction already freed — a
 * textbook cross-domain-pointer UAF (LMDB txn/page domain vs Lua GC domain).
 *
 * Detection note: for an ordinary small value LMDB *pools* the freed dirty page
 * onto env->me_dpages (mdb_page_free), so ASan would not see a free() — exactly
 * the libdbus-pool situation. We therefore store a MULTI-PAGE value: LMDB puts
 * big values in an overflow page, and mdb_dpage_free() free()s a multi-page
 * overflow buffer DIRECTLY (mdb.c: `!IS_OVERFLOW || mp_pages==1` -> pool, else
 * free). So the borrowed pointer lands in a buffer LMDB really free()s at txn
 * end, and the deferred read traps as a clean heap-use-after-free.
 *
 * Not for production. No MDB_WRITEMAP (so dirty pages are malloc'd, ASan-visible).
 */
#include <lua.h>
#include <lauxlib.h>
#include <lmdb.h>
#include <stdlib.h>

#define ENV_MT "minilmdb.env"
#define TXN_MT "minilmdb.txn"
#define VAL_MT "minilmdb.val"

typedef struct { MDB_env *env; MDB_dbi dbi; int dbi_open; } Env;
typedef struct { MDB_txn *txn; MDB_dbi dbi; int live; }    Txn;
typedef struct { const unsigned char *data; size_t size; } Val; /* BORROWED into a txn page */

static int fail(lua_State *L, const char *what, int rc) {
  return luaL_error(L, "%s: %s", what, mdb_strerror(rc));
}

/* minilmdb.open(path) -> env   (single-file DB via MDB_NOSUBDIR, no WRITEMAP) */
static int l_open(lua_State *L) {
  const char *path = luaL_checkstring(L, 1);
  int rc;
  Env *e = (Env *)lua_newuserdata(L, sizeof *e);
  e->dbi_open = 0; e->env = NULL;
  luaL_getmetatable(L, ENV_MT); lua_setmetatable(L, -2);
  if ((rc = mdb_env_create(&e->env)))               return fail(L, "env_create", rc);
  if ((rc = mdb_env_set_mapsize(e->env, 64u << 20))) return fail(L, "set_mapsize", rc);
  if ((rc = mdb_env_open(e->env, path, MDB_NOSUBDIR, 0664))) return fail(L, "env_open", rc);
  return 1;
}

/* env:begin() -> write txn (opens the main DBI lazily, in-txn) */
static int l_begin(lua_State *L) {
  Env *e = (Env *)luaL_checkudata(L, 1, ENV_MT);
  int rc;
  Txn *t = (Txn *)lua_newuserdata(L, sizeof *t);
  t->live = 0; t->txn = NULL;
  luaL_getmetatable(L, TXN_MT); lua_setmetatable(L, -2);
  if ((rc = mdb_txn_begin(e->env, NULL, 0, &t->txn))) return fail(L, "txn_begin", rc);
  if (!e->dbi_open) {
    if ((rc = mdb_dbi_open(t->txn, NULL, MDB_CREATE, &e->dbi))) return fail(L, "dbi_open", rc);
    e->dbi_open = 1;
  }
  t->dbi = e->dbi; t->live = 1;
  return 1;
}

/* txn:put(key, value) */
static int l_put(lua_State *L) {
  Txn *t = (Txn *)luaL_checkudata(L, 1, TXN_MT);
  size_t kl, vl; int rc;
  const char *k = luaL_checklstring(L, 2, &kl);
  const char *v = luaL_checklstring(L, 3, &vl);
  MDB_val K = { kl, (void *)k }, V = { vl, (void *)v };
  if ((rc = mdb_put(t->txn, t->dbi, &K, &V, 0))) return fail(L, "put", rc);
  return 0;
}

/* txn:get(key) -> value handle (NO COPY — borrows the pointer into the page) */
static int l_get(lua_State *L) {
  Txn *t = (Txn *)luaL_checkudata(L, 1, TXN_MT);
  size_t kl; int rc;
  const char *k = luaL_checklstring(L, 2, &kl);
  MDB_val K = { kl, (void *)k }, V;
  rc = mdb_get(t->txn, t->dbi, &K, &V);
  if (rc == MDB_NOTFOUND) { lua_pushnil(L); return 1; }
  if (rc) return fail(L, "get", rc);
  Val *val = (Val *)lua_newuserdata(L, sizeof *val);
  val->data = (const unsigned char *)V.mv_data;  /* <-- the bug: borrow, never copy */
  val->size = V.mv_size;
  luaL_getmetatable(L, VAL_MT); lua_setmetatable(L, -2);
  return 1;
}

static int l_commit(lua_State *L) {
  Txn *t = (Txn *)luaL_checkudata(L, 1, TXN_MT);
  if (t->live) { mdb_txn_commit(t->txn); t->live = 0; } /* frees dirty pages */
  return 0;
}
static int l_abort(lua_State *L) {
  Txn *t = (Txn *)luaL_checkudata(L, 1, TXN_MT);
  if (t->live) { mdb_txn_abort(t->txn); t->live = 0; }  /* frees dirty pages */
  return 0;
}

/* val:read() -> string. Dereferences the borrowed pointer NOW. If the txn has
 * ended, every load below hits the freed overflow page: ASan traps here (an
 * instrumented byte loop, so the first freed byte is the reported access). */
static int l_val_read(lua_State *L) {
  Val *v = (Val *)luaL_checkudata(L, 1, VAL_MT);
  luaL_Buffer b;
  char *out = luaL_buffinitsize(L, &b, v->size);
  for (size_t i = 0; i < v->size; i++) out[i] = (char)v->data[i]; /* freed-page load */
  luaL_pushresultsize(&b, v->size);
  return 1;
}

static const luaL_Reg env_m[] = { { "begin", l_begin }, { NULL, NULL } };
static const luaL_Reg txn_m[] = {
  { "put", l_put }, { "get", l_get }, { "commit", l_commit }, { "abort", l_abort }, { NULL, NULL }
};
static const luaL_Reg val_m[] = { { "read", l_val_read }, { NULL, NULL } };

static void mk_mt(lua_State *L, const char *name, const luaL_Reg *methods) {
  luaL_newmetatable(L, name);
  lua_pushvalue(L, -1); lua_setfield(L, -2, "__index");
  luaL_setfuncs(L, methods, 0);
  lua_pop(L, 1);
}

int luaopen_minilmdb(lua_State *L) {
  mk_mt(L, ENV_MT, env_m);
  mk_mt(L, TXN_MT, txn_m);
  mk_mt(L, VAL_MT, val_m);
  lua_newtable(L);
  lua_pushcfunction(L, l_open); lua_setfield(L, -2, "open");
  return 1;
}
