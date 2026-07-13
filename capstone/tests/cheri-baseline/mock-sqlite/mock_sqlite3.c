/* Minimal SQLite lifecycle harness — implementation. See sqlite3.h in this dir.
 *
 * Faithfulness rules (what each call does to memory, matching real SQLite so the
 * corpus defect manifests identically at the capability level):
 *   open        -> malloc a db handle.
 *   prepare     -> malloc a stmt, owned by the db (counts as "open").
 *   close       -> SQLITE_BUSY (do NOT free) if any stmt is still open, else free.
 *   close_v2    -> free the db handle (stmts kept, matching zombie-defer).
 *   exec        -> invoke the registered progress handler / UDF / authorizer
 *                  (each corpus shim registers exactly one), reproducing the
 *                  callback-context use-after-free rows.
 *   column_name -> return a stmt-owned heap buffer; finalize frees it (so a
 *                  cached column pointer dangles).
 *   finalize    -> free the stmt (and its column buffer); a second finalize on
 *                  the same handle is a genuine double-free.
 *   step/reset  -> dereference the stmt (so a stale stmt handle faults/reads).
 * No SQL is parsed or executed; this is not a database.
 */
#include "sqlite3.h"
#include <stdlib.h>
#include <string.h>

struct sqlite3 {
  int open_stmts;
  int alive;
  int (*progress_cb)(void *);
  void *progress_arg;
  void (*udf)(sqlite3_context *, int, sqlite3_value **);
  void *udf_userdata;
  int (*authorizer)(void *, int, const char *, const char *, const char *,
                    const char *);
  void *auth_arg;
};

struct sqlite3_stmt {
  sqlite3 *db;
  char *colname;   /* heap buffer returned by sqlite3_column_name */
  int value;
};

struct sqlite3_context {
  void *user_data;
};

int sqlite3_open(const char *filename, sqlite3 **ppDb) {
  (void)filename;
  sqlite3 *db = calloc(1, sizeof *db);
  if (!db) { *ppDb = 0; return SQLITE_ERROR; }
  db->alive = 1;
  *ppDb = db;
  return SQLITE_OK;
}

int sqlite3_close(sqlite3 *db) {
  if (!db) return SQLITE_OK;
  if (db->open_stmts > 0) return SQLITE_BUSY;   /* real sqlite3_close semantics */
  free(db);
  return SQLITE_OK;
}

int sqlite3_close_v2(sqlite3 *db) {
  /* close_v2 tears the connection down even with open stmts (deferred zombie
   * in real SQLite); the corpus UAFs are on the host wrappers, not the db. */
  free(db);
  return SQLITE_OK;
}

int sqlite3_prepare_v2(sqlite3 *db, const char *sql, int nByte,
                       sqlite3_stmt **ppStmt, const char **pzTail) {
  (void)sql; (void)nByte; (void)pzTail;
  sqlite3_stmt *st = calloc(1, sizeof *st);
  if (!st) { *ppStmt = 0; return SQLITE_ERROR; }
  st->db = db;
  st->value = 123;
  if (db) db->open_stmts++;
  *ppStmt = st;
  return SQLITE_OK;
}

int sqlite3_step(sqlite3_stmt *st) {
  /* A real step touches the statement AND its owning connection's internals
   * (VDBE reads db->pVdbe, db->mutex, ...). After the connection is closed this
   * is a use-after-free of the connection, which is the corpus defect. */
  volatile int touch = st->value;
  volatile int live = st->db->alive;   /* deref the (possibly freed) connection */
  (void)touch; (void)live;
  return SQLITE_ROW;
}

int sqlite3_reset(sqlite3_stmt *st) {
  volatile int live = st->db->alive;   /* reset also reaches into the connection */
  (void)live;
  return SQLITE_OK;
}

const char *sqlite3_column_name(sqlite3_stmt *st, int N) {
  (void)N;
  if (!st->colname) {
    st->colname = malloc(16);
    if (st->colname) memcpy(st->colname, "colname", 8);
  }
  return st->colname;   /* points into stmt-owned memory freed by finalize */
}

int sqlite3_column_int(sqlite3_stmt *st, int iCol) {
  (void)iCol;
  return st->value;
}

const unsigned char *sqlite3_column_text(sqlite3_stmt *st, int iCol) {
  (void)iCol;
  return (const unsigned char *)sqlite3_column_name(st, 0);
}

int sqlite3_finalize(sqlite3_stmt *st) {
  if (!st) return SQLITE_OK;
  if (st->db) st->db->open_stmts--;
  free(st->colname);     /* the cached column pointer now dangles */
  free(st);              /* a second finalize on this handle is a double-free */
  return SQLITE_OK;
}

int sqlite3_exec(sqlite3 *db, const char *sql, sqlite3_callback cb, void *arg,
                 char **errmsg) {
  (void)sql; (void)cb; (void)arg;
  if (errmsg) *errmsg = 0;
  if (!db) return SQLITE_ERROR;
  /* Reproduce statement execution invoking the registered callbacks. Each
   * corpus shim registers exactly one, so invoking all present is faithful. */
  if (db->progress_cb) {
    int (*p)(void *) = db->progress_cb;
    void *pa = db->progress_arg;
    p(pa);   /* row 1: the progress handler frees its context here */
  }
  if (db->udf) {
    struct sqlite3_context ctx;
    ctx.user_data = db->udf_userdata;
    db->udf(&ctx, 0, 0);   /* rows 2/6: the UDF dereferences a freed pApp */
  }
  if (db->authorizer) {
    db->authorizer(db->auth_arg, 0, "", "", "", "");  /* row 16 */
  }
  return SQLITE_OK;
}

void sqlite3_progress_handler(sqlite3 *db, int nOps, int (*cb)(void *),
                              void *arg) {
  if (!db) return;
  if (nOps > 0 && cb) { db->progress_cb = cb; db->progress_arg = arg; }
  else { db->progress_cb = 0; db->progress_arg = 0; }
}

int sqlite3_create_function(sqlite3 *db, const char *zName, int nArg,
                            int eTextRep, void *pApp,
                            void (*xFunc)(sqlite3_context *, int, sqlite3_value **),
                            void (*xStep)(sqlite3_context *, int, sqlite3_value **),
                            void (*xFinal)(sqlite3_context *)) {
  (void)zName; (void)nArg; (void)eTextRep; (void)xStep; (void)xFinal;
  if (!db) return SQLITE_ERROR;
  db->udf = xFunc;
  db->udf_userdata = pApp;
  return SQLITE_OK;
}

void *sqlite3_user_data(sqlite3_context *ctx) { return ctx->user_data; }
void  sqlite3_result_int(sqlite3_context *ctx, int v) { (void)ctx; (void)v; }
void  sqlite3_free(void *p) { free(p); }

int sqlite3_set_authorizer(sqlite3 *db,
        int (*xAuth)(void *, int, const char *, const char *, const char *,
                     const char *),
        void *pUserData) {
  if (!db) return SQLITE_ERROR;
  db->authorizer = xAuth;
  db->auth_arg = pUserData;
  return SQLITE_OK;
}

sqlite3_backup *sqlite3_backup_init(sqlite3 *pDest, const char *zDestName,
                                    sqlite3 *pSource, const char *zSourceName) {
  (void)zDestName; (void)zSourceName;
  /* backup reads both connections' schema/page cache; a closed source is a UAF */
  volatile int s = pSource->alive;
  volatile int d = pDest->alive;
  (void)s; (void)d;
  return calloc(1, sizeof(int));
}

int sqlite3_backup_finish(sqlite3_backup *p) { free(p); return SQLITE_OK; }
