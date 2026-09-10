/* row9 -- the LITERAL matched pair for
 * cve-repros/row9_ruby_finalize_after_dbfree (HIERARCHICAL-REVOKE), Ruby binding.
 *
 * The SAME row9 program as before-faithful.c, real SQLite C API, one Capstone
 * domain. sqlite3-ruby issue #49: a Statement outlives its Database; the Database
 * teardown finalizes the parent's tracked statements, but the child Statement
 * wrapper keeps a non-NULL, now-dangling `st`; its own #close later runs a SECOND
 * sqlite3_finalize on the freed statement -> use-after-free inside finalize.
 *
 * Faithful wrapper model (ext/sqlite3/{database,statement}.c):
 *   sqlite3Ruby     { sqlite3 *db;  ... }              // Database wrapper (parent)
 *   sqlite3StmtRuby { sqlite3_stmt *st; int done_p; }  // Statement wrapper (child)
 * REQUIRE_OPEN_STMT only checks `st != NULL`, so the stale handle passes the guard.
 *
 * Mechanism (revoke_on_free_hier_alloc.h): the Database gets its own SUB-ARENA,
 * MREV'd to a senior revocation node; the sqlite3 object AND the Vdbe statement
 * prepared on it are SPLIT DESCENDANTS of that senior node. The Database teardown
 * (clear_cache! / close) REVOKEs the senior node, sweeping the whole sub-lineage
 * -- the connection AND its tracked child statement -- in one operation. That is
 * the Ruby Database's ownership tree invalidating its children. The child wrapper
 * still holds `st`; its REQUIRE_OPEN_STMT NULL check passes; sqlite3_finalize(st)
 * then dereferences SQLite's own revoked statement handle and FAULTS.
 *
 *   host    : the child's 2nd sqlite3_finalize is a use-after-free of a stmt whose
 *             parent Database was torn down (ASan, inside sqlite3_finalize).
 *   Capstone: the same finalize on the hierarchically-revoked child is a
 *             deterministic fault.
 *
 * The pointer that faults is the exact sqlite3_stmt* sqlite3_prepare_v2 handed
 * back -- SQLite's own statement handle, revoked because its parent Database's
 * subtree was revoked. No wrapper around the handle, no carved copy.
 *
 * WHY THE REVOKE IS DRIVER-FIRED AT close (reported honestly, same as row7):
 * sqlite3_close_v2 is a ZOMBIE close -- with a live statement it does NOT free the
 * statement's memory, so there is no xFree for the flat allocator to turn into a
 * revoke. The cascade comes from the capability tree this domain builds (revoke
 * the Database's senior node at its wrapper teardown), modelling a binding whose
 * Database wrapper owns and tears down its statement wrappers. The scoping
 * property (a sibling Database survives one Database's close) is proven both at
 * the primitive level by tests/runtime-qemu/hier-revoke-probe and, for real
 * SQLite, by the ROW9_SIBLING variant below. Intra-domain: SPLIT + MREV + REVOKE.
 *
 * Cause is opt-level dependent (task-007/008): -O0 spills the statement handle
 * across the close so the reload clears the tag -> cause 24 (ROW9_NO_REVOKE is the
 * control); -O1/-O2 keep it register-held -> cause 25, self-proving.
 * See run-sqlite-row9.sh.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"
#include "revoke_on_free_hier_alloc.h"

#define CAPSTONE_DPI_REGION_SHARE 1U

/* Sub-arena sizes carved off the one linear grant (mirrors row7). The global
 * sub-arena holds connection-independent SQLite state (config/initialize); each
 * Database sub-arena holds one :memory: connection plus its statement. */
#ifndef ROW9_GLOBAL_ARENA
#define ROW9_GLOBAL_ARENA (512UL * 1024UL)
#endif
#ifndef ROW9_CONN_ARENA
#define ROW9_CONN_ARENA (1536UL * 1024UL)
#endif

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

/* When set, db_teardown does NOT fire the subtree revoke: the control. SQLite's
 * zombie close leaves the statement usable, so the child's finalize succeeds and
 * the domain RETURNS -- the -O0 cause-24 disambiguation control and the proof
 * that SQLite runs on the hierarchical allocator. */
static int row9_no_revoke;

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

/* ---- SQLite memory methods backed by the (hierarchical) flat allocator ---- */
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
  output_text("row9 SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  return rc ? rc : 1;
}

/* --- faithful subset of sqlite3-ruby's C glue (field names preserved) --- */
typedef struct {
  sqlite3_stmt *st; /* the child's retained statement handle */
  int done_p;
} sqlite3StmtRuby;

typedef struct {
  sqlite3 *db;             /* Database wrapper's connection */
  sqlite3StmtRuby *child;  /* one tracked statement (the @statements cache) */
} sqlite3Ruby;

#define REQUIRE_OPEN_STMT(ctx)                                                  \
  do {                                                                          \
    if (!(ctx)->st)                                                            \
      return 0;                                                                 \
  } while (0) /* Ruby: raises "closed statement"; only checks st != NULL */

static hconn conn_global;
static hconn conn_a;
static hconn conn_b; /* sibling Database (ROW9_SIBLING scoping proof) */

/* Statement#close: the child's own teardown. Its guard only checks st != NULL,
 * so a stale-but-non-NULL handle passes and finalize runs on it. */
static int stmt_rb_close(sqlite3StmtRuby *ctx) {
  REQUIRE_OPEN_STMT(ctx);
  int rc = sqlite3_finalize(ctx->st); /* on the revoked child -> FAULTS */
  ctx->st = (sqlite3_stmt *)0;
  return rc;
}

/* Database teardown (clear_cache! / db close). In the real bug this finalizes the
 * parent's tracked statements, freeing their sqlite3_stmt while the child wrapper
 * keeps a dangling handle. Here the Database wrapper's teardown REVOKEs its
 * sub-arena's senior node, sweeping the connection AND its tracked child
 * statement -- the ownership tree invalidating its children. */
static void db_teardown(sqlite3Ruby *db_obj, hconn *c) {
  hier_activate(c);
  sqlite3_close_v2(db_obj->db); /* zombie close: does NOT free the live stmt */
  db_obj->db = (sqlite3 *)0;
  hier_deactivate(c);
  if (!row9_no_revoke)
    hier_close(c); /* revoke the subtree: the tracked child stmt goes with it */
}

/* Open a Database on sub-arena `c` and prepare one tracked statement; both become
 * SPLIT descendants of c->rev. */
static int ruby_open(sqlite3Ruby *db_obj, sqlite3StmtRuby *stmt, hconn *c,
                     unsigned long arena) {
  hier_open(c, arena);
  hier_activate(c);
  int rc = sqlite3_open(":memory:", &db_obj->db);
  if (rc != SQLITE_OK) {
    hier_deactivate(c);
    return fail("open", rc);
  }
  rc = sqlite3_prepare_v2(db_obj->db, "SELECT 1", -1, &stmt->st, 0);
  hier_deactivate(c);
  if (rc != SQLITE_OK)
    return fail("prepare", rc);
  if (!stmt->st)
    return fail("prepare-null", 1);
  db_obj->child = stmt;
  return SQLITE_OK;
}

/* The literal row9 sequence. Returns only if the child's post-teardown finalize
 * does NOT fault (the hierarchical use-after-free we demonstrate is caught). */
static int run_row9(void) {
  hier_open(&conn_global, ROW9_GLOBAL_ARENA);
  hier_activate(&conn_global);
  int rc = sqlite3_config(SQLITE_CONFIG_MALLOC, &rof_mem_methods);
  if (rc != SQLITE_OK)
    return fail("config-malloc", rc);
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK)
    return fail("initialize", rc);
  hier_deactivate(&conn_global);

  sqlite3Ruby db_obj = {0};
  sqlite3StmtRuby stmt = {0};
  rc = ruby_open(&db_obj, &stmt, &conn_a, ROW9_CONN_ARENA);
  if (rc != SQLITE_OK)
    return rc;

#if defined(ROW9_SIBLING)
  /* Scoping proof: a SECOND Database `conn_b` is an independent SPLIT off the
   * main arena, not a descendant of conn_a->rev. Tear down conn_a (revoke), then
   * use conn_b's child statement -- it must SURVIVE and step/finalize cleanly. */
  sqlite3Ruby db_obj_b = {0};
  sqlite3StmtRuby stmt_b = {0};
  rc = ruby_open(&db_obj_b, &stmt_b, &conn_b, ROW9_CONN_ARENA);
  if (rc != SQLITE_OK)
    return rc;
  output_text("row9 prepared two sibling databases ok\n");

  db_teardown(&db_obj, &conn_a); /* revoke conn_a's subtree */
  output_text("row9 tore down database A\n");

  rc = sqlite3_step(stmt_b.st); /* sibling child: must NOT fault */
  output_text("row9 SIBLING survived close rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  (void)stmt_rb_close(&stmt_b);
  return 0;
#else
  output_text("row9 prepared child statement ok\n");

  /* before-faithful.c:  db_teardown(db_obj);  stmt_rb_close(stmt);
   * The Database teardown revokes conn_a's subtree (connection + child stmt).
   * The child wrapper still holds a non-NULL `st`. */
  db_teardown(&db_obj, &conn_a);
  output_text("row9 tore down database\n");

  /* The child #close: REQUIRE_OPEN_STMT passes (st != NULL), and
   * sqlite3_finalize(st) dereferences SQLite's own revoked statement handle and
   * FAULTS. In the control (row9_no_revoke) the zombie close leaves the statement
   * usable, so the finalize returns and the domain RETURNS. */
  rc = stmt_rb_close(&stmt);
  output_text("row9 NOTRAP child finalize rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  return 0;
#endif
}

void domain_main(void *arg, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (shared_region_count == 0)
      hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)arg;
    else if (shared_region_count == 1)
      hostcall_payload = (volatile char *)arg;
    else if (shared_region_count == 2)
      hier_init(arg); /* the LINEAR grant becomes the hierarchical main arena */
    ++shared_region_count;
    return;
  }

  if (hostcall_metadata)
    hostcall_metadata->length = 0;

#if defined(ROW9_NO_REVOKE)
  row9_no_revoke = 1;
#endif
  (void)run_row9();

  unsigned *res = (unsigned *)arg;
  *res = SQLITE_HC_RET_DONE;
}
