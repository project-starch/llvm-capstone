/* FZ12 — unbounded recursion (stack overflow) on a self-referential fts5vocab table.
 *
 * Collected by replaying SQLite's post-3.22.0 fuzz corpus against 3.22.0. Host ASan on
 * stock 3.22.0 reports a stack-overflow unwinding through the expression deleter:
 *
 *   #0 __interceptor_free
 *   #1 sqlite3MemFree       sqlite3.c:21551
 *   #2 sqlite3_free         sqlite3.c:25461
 *   #3 sqlite3DbFreeNN      sqlite3.c:25504
 *   #4 sqlite3ExprDeleteNN  sqlite3.c:93756
 *   #5 sqlite3ExprDelete    sqlite3.c:93760
 *
 * i.e. the recursion is in name resolution and the stack is already exhausted by the time
 * the parse tree is torn down -- the free path is where it finally tips over, not the cause.
 *
 * Minimised from a 277-byte fuzz case to 86 bytes:
 *
 *   CREATE VIRTUAL TABLE rowid USING fts5vocab( rowid , 'instance');
 *   SELECT * FROM rowid;
 *
 * fts5vocab's first argument names the FTS5 table it reads, so this declares a vocab table
 * that reads ITSELF. Resolving it recurses without a depth limit.
 *
 * The name `rowid` is NOT incidental, and this was verified: the same self-reference under
 * an ordinary name (CREATE VIRTUAL TABLE v USING fts5vocab(v,'col'); SELECT * FROM v)
 * runs CLEAN. `rowid` is special in the resolver, which is what sends it down the
 * recursive path instead of a clean "no such fts5 table" error.
 *
 * This is a denial-of-service bug, not a heap memory-safety bug -- it is in the corpus as
 * one, and should not be counted with the spatial set.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

static void kv(const char *k, long v){
  out_text("fz12 "); out_text(k); out_text("=");
  if(v < 0){ out_text("-"); v = -v; }
  out_uint((unsigned long)v); out_text("\n");
}

static int run_case(void){
  sqlite3 *db = 0; char *e = 0; int rc;
  if (repro_init()) return 1;

  rc = sqlite3_open(":memory:", &db);
  kv("open_rc", rc);
  if(rc != SQLITE_OK){ sqlite3_close(db); return 1; }

  /* REACHABILITY PROBE 1. The module must register, or nothing below means anything. */
  rc = sqlite3_exec(db,
        "CREATE VIRTUAL TABLE rowid USING fts5vocab( rowid , 'instance');", 0, 0, &e);
  kv("create_rc", rc);
  if(e){ out_text("fz12 create_err: "); out_text(e); out_text("\n"); sqlite3_free(e); e = 0; }
  if(rc != SQLITE_OK){
    out_text("fz12 the self-referential vocab table was REJECTED; bug site not reached\n");
    sqlite3_close(db); out_text("fz12 NOTRAP done\n"); return 0;
  }

  /* REACHABILITY PROBE 2. A clean "no such fts5 table" error here would mean the resolver
   * took the terminating path and the recursion never started -- a real negative, not a
   * silent pass. Anything else means we went down the recursive path. */
  out_text("fz12 about to resolve the self-reference (recursion starts here)\n");
  rc = sqlite3_exec(db, "SELECT * FROM rowid;", 0, 0, &e);
  kv("select_rc", rc);
  if(e){ out_text("fz12 select_err: "); out_text(e); out_text("\n"); sqlite3_free(e); e = 0; }

  sqlite3_close(db);
  out_text("fz12 NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("fz12")
