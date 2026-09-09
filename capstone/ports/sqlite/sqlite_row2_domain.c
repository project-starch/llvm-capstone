/* row2 -- the LITERAL matched pair for
 * cve-repros/row2_rusqlite_hook_closure_uaf (SEALED-CALLBACK / UAF of the
 * callback context).
 *
 * The SAME row2 program as before.c, real SQLite C API, one Capstone domain:
 *
 *     open :memory: -> create_function("read_context", pApp=app)
 *     -> free(app)  -> exec "SELECT read_context()"
 *
 * before.c registers a SQL function whose context pointer `app` is freed while
 * the function stays registered; when SQLite later invokes the callback it
 * dereferences the stale `app` via sqlite3_user_data (ASan heap-use-after-free,
 * RUSTSEC-2021-0128). Here `app` is an rof_malloc allocation (SQLite's whole heap
 * is the revoke-on-free linear allocator, as in row3 B2). Freeing it REVOKEs its
 * node. SQLite stores `app` as the function's pUserData and hands it back through
 * sqlite3_user_data on every invocation; after the free that stored pointer is
 * revoked, so when SQLite -- not the driver -- invokes read_context during
 * sqlite3_exec, the callback's dereference of the context FAULTS.
 *
 *   host    : the callback reads a freed context pointer (before.c).
 *   Capstone: the callback's read of the revoked context is a deterministic
 *             capability fault, and SQLite itself drives the invocation.
 *
 * The pointer that faults is the exact context SQLite returns from
 * sqlite3_user_data -- SQLite's own stored pApp, revoked by the host's own free.
 * No wrapper, no carved copy, no driver-fired revoke of anything but the freed
 * context.
 *
 * MECHANISM QUESTION (reported honestly): a "sealed callback" has two halves.
 *   (1) The CONTEXT lifetime -- the pApp UAF -- is what this domain demonstrates
 *       faithfully: real SQLite invokes the real registered callback and it
 *       faults on the revoked context. This is the actual memory-safety defect
 *       in rows 2/6/16.
 *   (2) The SEAL proper -- a sealed capability guarding a DOMAIN-CROSSING call
 *       boundary -- is NOT exercised here: SQLite invokes the callback through an
 *       ordinary C function pointer WITHIN this one domain, so no domain crossing
 *       and no unseal happens. A fully-faithful sealed-callback would require
 *       SQLite's callback dispatch to cross into a separate callback domain
 *       (__seal/__domcall on that boundary), which a single domain cannot
 *       express. See the history note and run-sqlite-row2.sh.
 *
 * Because SQLite keeps pApp in its function table (FuncDef.pUserData) and
 * reloads it on each call, the revoked context is always reached through a
 * MEMORY reload -- so the fault is cause 24 (tag gone on reload) at every opt
 * level, not the register-held cause 25. The no-revoke control disambiguates it.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"
#include "revoke_on_free_alloc.h"

#define CAPSTONE_DPI_REGION_SHARE 1U

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

static void output_text(const char *text) {
  if (!hostcall_metadata || !hostcall_payload)
    return;
  const char *src = (const char *)__builtin_capstone_cap_delin((void *)text);
  char *payload = (char *)__builtin_capstone_cap_delin((void *)hostcall_payload);
  unsigned long offset = hostcall_metadata->length;
  while (*src && offset + 1 < SQLITE_HC_REGION_SIZE)
    payload[offset++] = *src++;
  hostcall_metadata->length = offset;
}

static void output_uint(unsigned long v) {
  char buf[21];
  unsigned i = 21;
  buf[--i] = '\0';
  if (v == 0)
    buf[--i] = '0';
  while (v && i) {
    buf[--i] = (char)('0' + (v % 10));
    v /= 10;
  }
  output_text(&buf[i]);
}

/* ---- SQLite memory methods backed by the revoke-on-free allocator ---- */
static void *rof_xMalloc(int n) { return rof_malloc((unsigned long)n); }
static void rof_xFree(void *p) { rof_free(p); }
static int rof_xSize(void *p) { return (int)rof_size(p); }
static int rof_xRoundup(int n) { return (int)rof_roundup((unsigned long)n); }
static int rof_xInit(void *unused) {
  (void)unused;
  return SQLITE_OK;
}
static void rof_xShutdown(void *unused) { (void)unused; }

static void *rof_xRealloc(void *p, int n) {
  if (!p)
    return rof_malloc((unsigned long)n);
  if (n <= 0) {
    rof_free(p);
    return (void *)0;
  }
  unsigned long oldsz = rof_size(p);
  unsigned long newsz = rof_roundup((unsigned long)n);
  void *np = rof_malloc((unsigned long)n);
  if (!np)
    return (void *)0;
  unsigned long copy = oldsz < newsz ? oldsz : newsz;
  rof_copy_caps(np, p, copy);
  rof_free(p);
  return np;
}

static const sqlite3_mem_methods rof_mem_methods = {
    rof_xMalloc, rof_xFree,  rof_xRealloc,  rof_xSize,
    rof_xRoundup, rof_xInit, rof_xShutdown, (void *)0};

static int fail(const char *stage, int rc) {
  output_text("row2 SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  return rc ? rc : 1;
}

/* before.c's read_context: the registered SQL function. It reads its context
 * through sqlite3_user_data -- SQLite's own stored pApp. After the context is
 * revoked, this dereference FAULTS. Reached only through SQLite's callback
 * dispatch (sqlite3_exec -> sqlite3_step -> the function pointer). */
static void read_context(sqlite3_context *sqlctx, int argc,
                         sqlite3_value **argv) {
  (void)argc;
  (void)argv;
  int value = *(volatile int *)sqlite3_user_data(sqlctx); /* revoked -> FAULT */
  sqlite3_result_int(sqlctx, value);
}

static int run_row2(void) {
  sqlite3 *db = 0;

  int rc = sqlite3_config(SQLITE_CONFIG_MALLOC, &rof_mem_methods);
  if (rc != SQLITE_OK)
    return fail("config-malloc", rc);
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK)
    return fail("initialize", rc);
  rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK)
    return fail("open", rc);

  /* before.c:  int *app = malloc(sizeof(*app)); *app = 42;
   * app is an rof allocation -- an independently revocable capability. */
  int *app = (int *)rof_malloc(sizeof(int));
  if (!app)
    return fail("malloc-app", 1);
  *app = 42;

  /* before.c:  sqlite3_create_function(db, "read_context", 0, ..., app, ...);
   * SQLite stores app as the function's pUserData. */
  rc = sqlite3_create_function(db, "read_context", 0, SQLITE_UTF8, app,
                               read_context, 0, 0);
  if (rc != SQLITE_OK)
    return fail("create-function", rc);

  /* before.c:  free(app);   -- the host frees the callback context while the
   * function stays registered. With revoke-on-free, app's node is REVOKED. */
  rof_free(app);

  output_text("row2 registered function, freed context\n");

  /* before.c:  sqlite3_exec(db, "SELECT read_context()", ...);
   * SQLite invokes read_context, which dereferences the revoked context and
   * FAULTS. In the no-revoke control the context survives, the function returns
   * 42, and exec succeeds. */
  char *errmsg = 0;
  rc = sqlite3_exec(db, "SELECT read_context()", 0, 0, &errmsg);
  output_text("row2 NOTRAP exec rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  (void)sqlite3_close(db);
  return 0;
}

void domain_main(void *arg, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (shared_region_count == 0)
      hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)arg;
    else if (shared_region_count == 1)
      hostcall_payload = (volatile char *)arg;
    else if (shared_region_count == 2)
      rof_init(arg); /* the LINEAR grant becomes the allocator's arena */
    ++shared_region_count;
    return;
  }

  if (hostcall_metadata)
    hostcall_metadata->length = 0;

#if defined(ROW2_NO_REVOKE)
  rof_no_revoke = 1;
#endif
  (void)run_row2();

  unsigned *res = (unsigned *)arg;
  *res = SQLITE_HC_RET_DONE;
}
