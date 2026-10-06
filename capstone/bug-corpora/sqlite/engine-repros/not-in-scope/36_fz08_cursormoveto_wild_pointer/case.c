/* FZ08 — SEGV in sqlite3VdbeCursorMoveto on a deeply nested CTE/sub-select.
 *
 * Collected by replaying SQLite's post-3.22.0 fuzz corpus against 3.22.0. Host ASan on
 * stock 3.22.0, SEGV on address 0x3 (a small non-zero address, i.e. a bad pointer rather
 * than a plain NULL):
 *
 *   #0 sqlite3VdbeCursorMoveto  sqlite3.c:76109    VdbeCursor *p = *pp;
 *   #1 sqlite3VdbeExec          sqlite3.c:82409
 *
 * The trigger is a single self-contained statement: a CTE named c whose body is a
 * VALUES/UNION, with min(-i) aggregates over further CTEs of the same name nested inside
 * sub-selects. The repeated shadowing of the name `c` at several nesting depths is what
 * drives the cursor bookkeeping wrong, so the nesting is NOT decoration and must be kept.
 *
 * Kept verbatim from the fuzz case (the inner text is machine-generated and resists
 * hand-minimisation; reducing the nesting makes it run clean).
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

/* The fuzz case used the float literals 5.1 and 3.261. This port builds with
 * -DSQLITE_OMIT_FLOATING_POINT=1, which puts the float-literal branch of sqlite3GetToken()
 * behind an #ifndef, so "5.1" tokenises as TK_INTEGER "5" plus a stray "." and prepare fails
 * with `near ".": syntax error` -- the probe below caught exactly that on the first run.
 * Verified on the host that the floats are NOT part of the mechanism: with 5.1 -> 5 and
 * 3.261 -> 3 the case still faults at sqlite3VdbeCursorMoveto, so the integers are used here
 * and no build-flag change is needed. */
static const char *zTrigger =
  "WITH c(i)AS(VALUES(8)UNIoN SELECT 5)SELECT min(- i )|(WITH c( c )AS(VALUES( i )"
  "UNIoN SELECT i LIKE 3 ESCAPE '' )SELECT min(- (SELECT min(- i IN(SELECT c ) )|"
  "(WITH c( c )AS(VALUES( i )UNIoN SELECT i )SELECT min(- i )=(SELECT min(- (SELECT "
  "min(- i )|(WITH c( c )AS(VALUES( i )UNIoN SELECT i )SELECT min(- i )=i fROM c)i fROM c)"
  " )=i fROM c) fROM c)i fROM c) )=i NOTNULL fROM c)i fROM c;";

static void kv(const char *k, long v){
  out_text("fz08 "); out_text(k); out_text("=");
  if(v < 0){ out_text("-"); v = -v; }
  out_uint((unsigned long)v); out_text("\n");
}

static int run_case(void){
  sqlite3 *db = 0; sqlite3_stmt *st = 0; int rc, nstep = 0;
  if (repro_init()) return 1;

  rc = sqlite3_open(":memory:", &db);
  kv("open_rc", rc);
  if(rc != SQLITE_OK){ sqlite3_close(db); return 1; }

  /* REACHABILITY PROBE. The fault is in sqlite3VdbeExec, i.e. at STEP time, not at
   * prepare time. prepare_rc must be 0 or the statement never reaches the VDBE and this
   * case establishes nothing. */
  rc = sqlite3_prepare_v2(db, zTrigger, -1, &st, 0);
  kv("prepare_rc", rc);
  if(rc != SQLITE_OK){
    out_text("fz08 prepare FAILED: "); out_text(sqlite3_errmsg(db)); out_text("\n");
    out_text("fz08 bug site is in sqlite3VdbeExec and was not reached\n");
    sqlite3_close(db); out_text("fz08 NOTRAP done\n"); return 0;
  }

  out_text("fz08 about to step (sqlite3VdbeCursorMoveto runs here)\n");
  while((rc = sqlite3_step(st)) == SQLITE_ROW && nstep < 64) nstep++;
  kv("step_rc", rc);
  kv("rows", nstep);

  sqlite3_finalize(st);
  sqlite3_close(db);
  out_text("fz08 NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("fz08")
