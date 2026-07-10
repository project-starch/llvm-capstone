/* row5 -- the LITERAL matched pair for
 * cve-repros/row5_php_destruction_order (HIERARCHICAL-REVOKE), PHP binding.
 *
 * The SAME row5 program as before-faithful.c, real SQLite C API, one Capstone
 * domain. PHP bug #69971: at request shutdown the Zend GC frees the DB object
 * BEFORE the statement object. The statement free handler then reaches back into
 * its parent connection (finalizing its statement, walking db_obj->free_list) --
 * but that connection was already torn down.
 *
 * Faithful wrapper model (ext/sqlite3/php_sqlite3_structs.h, field names kept):
 *   php_sqlite3_db_object   { sqlite3 *db;  zend_llist free_list; ... }  // parent
 *   php_sqlite3_stmt_object { sqlite3_stmt *stmt;  db_object *db_obj; }  // child,
 *                                            with a back-pointer to the parent.
 * The wrong destruction order (db object before stmt object) is preserved.
 *
 * Mechanism (revoke_on_free_hier_alloc.h): the connection gets its own SUB-ARENA,
 * MREV'd to a senior revocation node. SQLite's own allocations for it -- the
 * sqlite3 connection AND the Vdbe statement -- are SPLIT DESCENDANTS of that
 * senior node. (The Zend wrapper objects model a SEPARATE heap and live on the
 * surviving global arena, exactly as the real bug frees them in the wrong order
 * rather than with the connection.) The DB object's free handler runs FIRST and
 * tears the connection down: it REVOKEs the senior node, sweeping the connection
 * and its child statement. The statement free handler then runs; its first
 * SQLite-owned access, sqlite3_finalize(stmt->stmt), dereferences SQLite's own
 * revoked statement handle and FAULTS -- the child statement swept with its
 * parent connection, one step before the real crash's db_obj->free_list read.
 *
 *   host    : the statement free handler touches the already-freed connection
 *             (ASan heap-use-after-free in php_sqlite3_compare_stmt_free).
 *   Capstone: sqlite3_finalize faults on the hierarchically-revoked child
 *             statement -- SQLite's own handle.
 *
 * The pointer that faults is the exact sqlite3_stmt* sqlite3_prepare_v2 handed
 * back -- SQLite's own statement handle, revoked because its parent connection's
 * subtree was revoked. No wrapper around the handle, no carved copy.
 *
 * WHY THE REVOKE IS DRIVER-FIRED AT the wrapper teardown (reported honestly, same
 * as row7/row9): sqlite3_close_v2 is a ZOMBIE close and does not free the live
 * statement, so the cascade comes from the capability tree this domain builds
 * (revoke the connection's senior node when its wrapper is destroyed), modelling
 * the Zend object graph whose db object owns its statement object. The scoping
 * property (a sibling connection survives) is proven at the primitive level by
 * tests/runtime-qemu/hier-revoke-probe and, for real SQLite, by ROW5_SIBLING.
 * Intra-domain: SPLIT + MREV + REVOKE only, no monitor change.
 *
 * Cause is opt-level dependent (task-007/008): -O0 spills the statement handle so
 * the reload clears the tag -> cause 24 (ROW5_NO_REVOKE is the control); -O1/-O2
 * keep it register-held -> cause 25, self-proving. See run-sqlite-row5.sh.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"
#include "revoke_on_free_hier_alloc.h"

#define CAPSTONE_DPI_REGION_SHARE 1U

#ifndef ROW5_GLOBAL_ARENA
#define ROW5_GLOBAL_ARENA (512UL * 1024UL)
#endif
#ifndef ROW5_CONN_ARENA
#define ROW5_CONN_ARENA (1536UL * 1024UL)
#endif

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

/* When set, the DB object free handler does NOT fire the subtree revoke: the
 * control. The connection wrapper stays intact, the statement free handler's
 * finalize + free_list walk both succeed, and the domain RETURNS. */
static int row5_no_revoke;

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
  output_text("row5 SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  return rc ? rc : 1;
}

/* --- faithful subset of ext/sqlite3/php_sqlite3_structs.h (field names kept) --- */
struct llist_node {
  void *data;
  struct llist_node *next;
};
typedef struct {
  struct llist_node *head;
  unsigned long count;
} zend_llist;

typedef struct php_sqlite3_db_object {
  int initialised;
  sqlite3 *db;
  zend_llist free_list; /* the connection tracks its live statements */
} php_sqlite3_db_object;

typedef struct {
  sqlite3_stmt *stmt;
  php_sqlite3_db_object *db_obj; /* back-pointer to the parent connection */
  int initialised;
} php_sqlite3_stmt_object;

static hconn conn_global;
static hconn conn_a;
static hconn conn_b; /* sibling connection (ROW5_SIBLING scoping proof) */

/* php_sqlite3_compare_stmt_free(): the real crash site. Walks the parent
 * connection's free_list to unlink itself -- a read of db_obj (revoked). */
static void php_sqlite3_compare_stmt_free(php_sqlite3_db_object *db_obj,
                                          php_sqlite3_stmt_object *stmt) {
  struct llist_node **pp = &db_obj->free_list.head; /* <-- read of revoked parent */
  while (*pp) {
    if ((*pp)->data == stmt) {
      *pp = (*pp)->next;
      db_obj->free_list.count--;
      break;
    }
    pp = &(*pp)->next;
  }
}

/* Zend free_storage for the DB object -- runs FIRST at shutdown. Tears the
 * connection wrapper down: close + REVOKE the connection's whole subtree. */
static void php_sqlite3_object_free_storage(php_sqlite3_db_object *db_obj,
                                            hconn *c) {
  hier_activate(c);
  if (db_obj->initialised)
    sqlite3_close_v2(db_obj->db); /* zombie close: does NOT free the live stmt */
  hier_deactivate(c);
  if (!row5_no_revoke)
    hier_close(c); /* subtree revoke: wrapper + connection + child stmt */
}

/* Zend free_storage for the STMT object -- runs AFTER the db object. Its first
 * parent-owned access (finalize the child stmt) faults on the revoked child;
 * the free_list walk that follows is the real crash line, on the same revoked
 * wrapper. */
static int php_sqlite3_stmt_object_free_storage(php_sqlite3_stmt_object *stmt) {
  int rc = 0;
  if (stmt->initialised && stmt->stmt)
    rc = sqlite3_finalize(stmt->stmt); /* on the revoked child -> FAULTS */
  php_sqlite3_compare_stmt_free(stmt->db_obj, stmt);
  return rc;
}

/* Open a connection whose SQLite allocations live on sub-arena `c`. The PHP/Zend
 * wrapper objects (db_object, stmt_object, the free_list node) model the ZEND
 * heap -- a separate allocator from SQLite's -- so they are carved from the
 * SURVIVING conn_global arena, NOT from `c`. Only the sqlite3 connection and the
 * Vdbe statement (SQLite's own allocations) become SPLIT descendants of c->rev.
 * That is the hierarchy under test: the connection (parent) owns the statement
 * (child); revoking the connection sweeps the child statement, while the Zend
 * wrapper bookkeeping stays readable (it is freed later, in the wrong order). */
static int php_open(php_sqlite3_db_object **db_out,
                    php_sqlite3_stmt_object **stmt_out, hconn *c,
                    unsigned long arena) {
  /* Zend-heap wrapper objects: allocate on the surviving global arena. */
  hier_activate(&conn_global);
  php_sqlite3_db_object *db_obj = rof_malloc(sizeof(*db_obj));
  php_sqlite3_stmt_object *stmt = rof_malloc(sizeof(*stmt));
  struct llist_node *n = rof_malloc(sizeof(*n));
  hier_deactivate(&conn_global);
  if (!db_obj || !stmt || !n)
    return fail("wrapper-malloc", 1);
  db_obj->initialised = 0;
  db_obj->db = (sqlite3 *)0;
  db_obj->free_list.head = (struct llist_node *)0;
  db_obj->free_list.count = 0;
  stmt->stmt = (sqlite3_stmt *)0;
  stmt->db_obj = db_obj;
  stmt->initialised = 0;

  /* SQLite's own allocations for this connection: on sub-arena `c`. */
  hier_open(c, arena);
  hier_activate(c);
  int rc = sqlite3_open(":memory:", &db_obj->db);
  if (rc != SQLITE_OK) {
    hier_deactivate(c);
    return fail("open", rc);
  }
  db_obj->initialised = 1;
  rc = sqlite3_prepare_v2(db_obj->db, "SELECT 1", -1, &stmt->stmt, 0);
  if (rc != SQLITE_OK) {
    hier_deactivate(c);
    return fail("prepare", rc);
  }
  if (!stmt->stmt) {
    hier_deactivate(c);
    return fail("prepare-null", 1);
  }
  hier_deactivate(c);
  stmt->initialised = 1;

  /* connection tracks the statement (llist_add) via the Zend-heap node. */
  n->data = stmt;
  n->next = db_obj->free_list.head;
  db_obj->free_list.head = n;
  db_obj->free_list.count++;

  *db_out = db_obj;
  *stmt_out = stmt;
  return SQLITE_OK;
}

/* The literal row5 sequence. Returns only if the post-teardown statement free
 * handler does NOT fault (the hierarchical use-after-free we demonstrate is
 * caught). */
static int run_row5(void) {
  hier_open(&conn_global, ROW5_GLOBAL_ARENA);
  hier_activate(&conn_global);
  int rc = sqlite3_config(SQLITE_CONFIG_MALLOC, &rof_mem_methods);
  if (rc != SQLITE_OK)
    return fail("config-malloc", rc);
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK)
    return fail("initialize", rc);
  hier_deactivate(&conn_global);

  php_sqlite3_db_object *db_obj = 0;
  php_sqlite3_stmt_object *stmt = 0;
  rc = php_open(&db_obj, &stmt, &conn_a, ROW5_CONN_ARENA);
  if (rc != SQLITE_OK)
    return rc;

#if defined(ROW5_SIBLING)
  /* Scoping proof: a SECOND connection `conn_b` is an independent SPLIT off the
   * main arena, not a descendant of conn_a->rev. Destroy conn_a's db object
   * (revoke), then finalize conn_b's statement -- it must SURVIVE. */
  php_sqlite3_db_object *db_obj_b = 0;
  php_sqlite3_stmt_object *stmt_b = 0;
  rc = php_open(&db_obj_b, &stmt_b, &conn_b, ROW5_CONN_ARENA);
  if (rc != SQLITE_OK)
    return rc;
  output_text("row5 prepared two sibling connections ok\n");

  php_sqlite3_object_free_storage(db_obj, &conn_a); /* revoke conn_a's subtree */
  output_text("row5 freed db object A\n");

  rc = sqlite3_step(stmt_b->stmt); /* sibling child: must NOT fault */
  output_text("row5 SIBLING survived free rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  (void)sqlite3_finalize(stmt_b->stmt);
  return 0;
#else
  output_text("row5 prepared child statement ok\n");

  /* before-faithful.c wrong order: DB object freed first, then the STMT object.
   * The DB object free handler revokes conn_a's subtree. */
  php_sqlite3_object_free_storage(db_obj, &conn_a);
  output_text("row5 freed db object\n");

  /* The STMT object free handler runs AFTER: its first SQLite-owned access,
   * sqlite3_finalize(stmt->stmt), dereferences SQLite's own revoked statement
   * handle and FAULTS inside finalize (the following db_obj->free_list read --
   * the real crash line -- is never reached). In the control (row5_no_revoke)
   * the connection is intact, so finalize and the free_list walk both succeed
   * and the domain RETURNS. */
  rc = php_sqlite3_stmt_object_free_storage(stmt);
  output_text("row5 NOTRAP stmt free handler rc=");
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

#if defined(ROW5_NO_REVOKE)
  row5_no_revoke = 1;
#endif
  (void)run_row5();

  unsigned *res = (unsigned *)arg;
  *res = SQLITE_HC_RET_DONE;
}
