/* row7 -- the LITERAL matched pair for
 * cve-repros/row7_cpython_cursor_dealloc (HIERARCHICAL-REVOKE).
 *
 * The SAME row7 program as before.c, real SQLite C API, one Capstone domain:
 *
 *     open :memory: -> prepare "SELECT 1" (statement is a CHILD of the
 *     connection) -> close_v2(connection) [+ free the cursor wrapper]
 *     -> step(statement)
 *
 * before.c closes the connection, frees the cursor wrapper, then steps the
 * statement -- a use-after-free of a statement whose parent connection was torn
 * down (ASan SEGV). This is the HIERARCHICAL shape: the statement's authority is
 * a CHILD of the connection's; destroying the parent must revoke the child.
 *
 * Mechanism (revoke_on_free_hier_alloc.h): the connection is given its own
 * SUB-ARENA, MREV'd to a senior revocation node; every SQLite allocation made
 * while that connection is active -- the sqlite3 object AND the Vdbe statement
 * -- is a SPLIT descendant of that senior node. Closing the connection REVOKEs
 * the senior node, sweeping the whole sub-lineage (connection + live statement)
 * in one operation. A later sqlite3_step on the child statement dereferences
 * revoked memory and FAULTS.
 *
 *   host    : step after close+free is a use-after-free of the statement.
 *   Capstone: step after the hierarchical revoke is a deterministic fault.
 *
 * The pointer that faults is the exact sqlite3_stmt* sqlite3_prepare_v2 handed
 * back -- SQLite's own statement handle, revoked because its parent connection's
 * subtree was revoked. No wrapper around the statement, no carved copy.
 *
 * WHY THE REVOKE IS DRIVER-FIRED AT close (reported honestly): SQLite's
 * sqlite3_close_v2 is a ZOMBIE close -- with a live statement it does NOT free
 * the statement's memory, so there is no xFree for the flat allocator to turn
 * into a revoke. The cascade therefore comes from the capability tree this
 * domain builds (revoke the connection's senior node at close), modelling a
 * binding whose connection wrapper owns and tears down its statement wrappers.
 * The scoping property (a sibling connection survives one connection's close) is
 * proven at the primitive level by tests/runtime-qemu/hier-revoke-probe
 * (hier_sibling_conn_survives_ok); this domain demonstrates the literal cascade
 * on real SQLite. Intra-domain: SPLIT + MREV + REVOKE only, no monitor change.
 *
 * Cause is opt-level dependent (task-007/008): -O0 spills the statement handle
 * across the close so the reload clears the tag -> cause 24 (ROW7_NO_REVOKE is
 * the control); -O1/-O2 keep it register-held -> cause 25, self-proving.
 * See run-sqlite-row7.sh.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"
#include "revoke_on_free_hier_alloc.h"

#define CAPSTONE_DPI_REGION_SHARE 1U

/* Sub-arena sizes carved off the one linear grant. The global arena holds
 * connection-independent SQLite state (sqlite3_config/initialize); the
 * connection arena holds one :memory: connection plus its statement. */
#ifndef ROW7_GLOBAL_ARENA
#define ROW7_GLOBAL_ARENA (512UL * 1024UL)
#endif
#ifndef ROW7_CONN_ARENA
#define ROW7_CONN_ARENA (3UL * 1024UL * 1024UL)
#endif

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

/* When set, hier_close recycles nothing and does NOT fire the revoke: the
 * control. SQLite's zombie close leaves the statement usable, so the post-close
 * step succeeds and the domain returns -- disambiguating the -O0 cause-24 fault
 * and proving SQLite runs on the hierarchical allocator. */
static int row7_no_revoke;

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
  output_text("row7 SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  return rc ? rc : 1;
}

static hconn conn_global;
static hconn conn_a;

/* The literal row7 sequence. Returns only if the post-close step does NOT
 * fault (the hierarchical use-after-free we demonstrate is caught). */
static int run_row7(void) {
  /* Global sub-arena for connection-independent SQLite state. Never closed. */
  hier_open(&conn_global, ROW7_GLOBAL_ARENA);
  hier_activate(&conn_global);

  int rc = sqlite3_config(SQLITE_CONFIG_MALLOC, &rof_mem_methods);
  if (rc != SQLITE_OK)
    return fail("config-malloc", rc);
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK)
    return fail("initialize", rc);
  hier_deactivate(&conn_global);

  /* Connection A gets its own sub-arena: everything SQLite allocates for this
   * connection -- the sqlite3 object and the Vdbe statement -- becomes a SPLIT
   * descendant of conn_a's senior revocation node. */
  sqlite3 *connection = 0;
  sqlite3_stmt *statement = 0;
  hier_open(&conn_a, ROW7_CONN_ARENA);
  hier_activate(&conn_a);

  rc = sqlite3_open(":memory:", &connection);
  if (rc != SQLITE_OK)
    return fail("open", rc);
  rc = sqlite3_prepare_v2(connection, "SELECT 1", -1, &statement, 0);
  if (rc != SQLITE_OK)
    return fail("prepare", rc);
  if (!statement)
    return fail("prepare-null", 1);
  hier_deactivate(&conn_a);

  output_text("row7 prepared child statement ok\n");

  /* before.c:  sqlite3_close_v2(cursor->connection);  free(cursor);
   * Zombie close: SQLite does NOT free the live statement. The connection
   * wrapper's teardown revokes conn_a's whole subtree -- the connection AND the
   * child statement. */
  hier_activate(&conn_a);
  rc = sqlite3_close_v2(connection);
  hier_deactivate(&conn_a);
  if (rc != SQLITE_OK)
    return fail("close", rc);

  if (!row7_no_revoke)
    hier_close(&conn_a); /* revoke the parent subtree (models wrapper destruction) */

  output_text("row7 closed connection\n");

  /* before.c:  sqlite3_step(cursor->statement);  -- USE AFTER the parent's
   * teardown. The child statement's memory was revoked with the connection's
   * subtree, so this dereference of SQLite's own statement handle FAULTS.
   * In the control (row7_no_revoke) SQLite's zombie semantics leave the
   * statement usable, so the step succeeds and we RETURN. */
  rc = sqlite3_step(statement);
  output_text("row7 NOTRAP post-close step rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  (void)sqlite3_finalize(statement);
  return 0;
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

#if defined(ROW7_NO_REVOKE)
  row7_no_revoke = 1;
#endif
  (void)run_row7();

  unsigned *res = (unsigned *)arg;
  *res = SQLITE_HC_RET_DONE;
}
