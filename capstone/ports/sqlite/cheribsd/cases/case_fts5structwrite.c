/* row 3 / sqlite-25e3073741 -- fts5 iterator caches an Fts5Structure that a concurrent
 * write releases.
 *
 * fts5MultiIterNew() stores a raw Fts5Structure* in Fts5Iter.pStruct; a write under an
 * active cursor releases that structure, so the pointer dangles, and fts5MultiIterFree()
 * releases it AGAIN. The 3.27.0 fix deletes the field outright -- it removes
 * `Fts5Structure *pStruct;` from Fts5Iter, the `pNew->pStruct = pStruct;` store and the
 * `fts5StructureRelease(pIter->pStruct)` call in fts5MultiIterFree.
 *
 * WHAT CHANGED HERE, AND WHY. An earlier version of this case matched
 * 'two OR three OR four' over four documents with DISTINCT terms and wrote once, on the
 * first row. That reached neither a shared structure nor a re-seek, so it ran to NOTRAP
 * establishing nothing. Upstream's own test is the shape that matters: five documents in
 * which ONE term repeats, a single-term MATCH, and a write on EVERY row of the scan --
 * which forces fts5CursorReseek, and the re-seek is where the freed structure is
 * released a second time.
 *
 * Host ASan oracle (3.22.0), upstream's fts5update.test 3.0/3.1 shape:
 *   READ of size 4 -- Fts5Structure.nRef, which is the struct's FIRST member
 *     use   fts5StructureRelease <- fts5MultiIterFree <- sqlite3Fts5IterClose
 *             <- fts5ExprNearInitAll <- fts5ExprNodeFirst <- sqlite3Fts5ExprFirst
 *             <- fts5CursorReseek <- fts5NextMethod
 * It fires with either an OPTIMIZE or a plain INSERT as the in-loop write; this case
 * uses OPTIMIZE, which is what upstream uses.
 *
 * Note nRef at offset 0 is exactly the granule memsys5's 8-byte Mem5Link overwrites, so
 * in the domain the refcount read returns a freelist integer rather than faulting -- see
 * case_mem5tagmap.c. That is why this is a silent control here and a reported UAF on the
 * host.
 *
 * REACHABILITY PROBE: the MATCH must return 3 rows ('one' appears in documents 1, 3 and
 * 5) and every in-loop write must succeed, which together prove the structure really was
 * replaced underneath a live cursor. The domain prints both and warns if either fails.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

static void rcline(const char *what, int rc){
  out_text("fts5structwrite "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; char *e=0; int rc;

  rc = sqlite3_open(":memory:", &db);
  rcline("open", rc);
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  /* upstream fts5update.test 3.0 verbatim: 'one' repeats across documents 1, 3 and 5 */
  rc = sqlite3_exec(db,
      "CREATE VIRTUAL TABLE x3 USING fts5(x);"
      "INSERT INTO x3 VALUES('one');"
      "INSERT INTO x3 VALUES('two');"
      "INSERT INTO x3 VALUES('one');"
      "INSERT INTO x3 VALUES('two');"
      "INSERT INTO x3 VALUES('one');", 0,0,&e);
  rcline("setup", rc);
  if(e){ out_text("fts5structwrite setup err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  /* upstream 3.1: write on EVERY row of the scan, which forces a cursor re-seek */
  sqlite3_stmt *st=0;
  rc = sqlite3_prepare_v2(db,"SELECT x FROM x3 WHERE x3 MATCH 'one'",-1,&st,0);
  rcline("scan prepare", rc);
  if(rc!=SQLITE_OK){
    out_text("fts5structwrite prepare err ("); out_text(sqlite3_errmsg(db)); out_text(")\n");
    sqlite3_close(db); return 1;
  }
  int n=0, nwrite=0, writefail=0;
  while(sqlite3_step(st)==SQLITE_ROW){
    n++;
    int wrc = sqlite3_exec(db,"INSERT INTO x3(x3) VALUES('optimize');",0,0,&e);
    if(e){ sqlite3_free(e); e=0; }
    if(wrc==SQLITE_OK) nwrite++; else { writefail++;
      if(writefail<=2){ out_text("fts5structwrite in-loop write rc=");
        out_uint((unsigned)(wrc<0?-wrc:wrc)); out_text("\n"); } }
  }
  int endrc = sqlite3_errcode(db);
  sqlite3_finalize(st);

  out_text("fts5structwrite rows="); out_uint((unsigned)n);
  out_text(" in_loop_writes_ok="); out_uint((unsigned)nwrite);
  out_text(" write_failures="); out_uint((unsigned)writefail);
  out_text(" end_rc="); out_uint((unsigned)(endrc<0?-endrc:endrc)); out_text("\n");
  out_text("fts5structwrite   (host oracle on this shape: rows=3 and it faults in"
           " fts5StructureRelease reading the freed nRef)\n");
  if(n!=3)
    out_text("fts5structwrite WARNING expected 3 rows; the repeated-term MATCH did not"
             " drive the scan, so no re-seek over a replaced structure happened\n");
  if(nwrite==0)
    out_text("fts5structwrite WARNING no in-loop write succeeded; the structure was"
             " never replaced under the live cursor\n");

  rc = sqlite3_exec(db,"INSERT INTO x3(x3) VALUES('integrity-check');",0,0,&e);
  rcline("integrity", rc);
  if(e){ sqlite3_free(e); e=0; }

  sqlite3_close(db);
  out_text("fts5structwrite NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts5structwrite")
