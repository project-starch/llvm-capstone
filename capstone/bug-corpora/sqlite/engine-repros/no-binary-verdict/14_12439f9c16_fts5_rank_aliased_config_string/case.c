/* row12 / sqlite-12439f9c16 -- fts5FilterMethod() parses rank from a config that
 * has NOT yet been reloaded. fts5CursorParseRank ALIASES the config strings:
 *     pCsr->zRank     = (char*)pConfig->zRank;
 *     pCsr->zRankArgs = (char*)pConfig->zRankArgs;
 * When another connection redefines rank, %_config changes and the cookie bumps;
 * the later sqlite3Fts5ConfigLoad does
 *     sqlite3_free(pConfig->zRank); sqlite3_free(pConfig->zRankArgs);
 * so the cursor's aliases dangle and the subsequent reads of pCsr->zRank are a
 * use-after-free. Fixed 3.43.2 (the fix hoists the reload above rank parsing).
 *
 * Needs TWO connections on ONE database, so this builds in the fts5S group with
 * -USQLITE_OMIT_SHARED_CACHE and opens file::memory:?cache=shared via
 * SQLITE_OPEN_URI. The VFS xOpen always returns SQLITE_CANTOPEN, but a
 * :memory: database (including the shared-cache URI form) takes the memDb path
 * and never calls xOpen.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

#define URI "file::memory:?cache=shared"

static int openshared(sqlite3 **pdb, const char *tag){
  int rc = sqlite3_open_v2(URI, pdb,
      SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|SQLITE_OPEN_URI, 0);
  out_text("fts5rank open "); out_text(tag); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc));
  if(rc!=SQLITE_OK && *pdb){ out_text(" ("); out_text(sqlite3_errmsg(*pdb)); out_text(")"); }
  out_text("\n");
  return rc;
}

static int run_case(void){
  if (repro_init()) return 1;
  int rc = sqlite3_enable_shared_cache(1);
  out_text("fts5rank shared_cache rc="); out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");

  sqlite3 *db1=0, *db2=0;
  if(openshared(&db1,"db1")) { sqlite3_close(db1); return 1; }
  if(openshared(&db2,"db2")) { sqlite3_close(db1); sqlite3_close(db2); return 1; }

  char *e=0;
  rc=sqlite3_exec(db1,
    "CREATE VIRTUAL TABLE ft USING fts5(a);"
    "INSERT INTO ft VALUES('alpha beta gamma');"
    "INSERT INTO ft VALUES('beta gamma delta');"
    "INSERT INTO ft VALUES('gamma delta epsilon');",0,0,&e);
  out_text("fts5rank setup rc="); out_uint((unsigned)(rc<0?-rc:rc));
  if(e){ out_text(" ("); out_text(e); out_text(")"); sqlite3_free(e); e=0; }
  out_text("\n");

  /* prime db1: load the fts5 config + learn the cookie */
  sqlite3_stmt *st=0;
  if(sqlite3_prepare_v2(db1,"SELECT a FROM ft WHERE ft MATCH 'gamma' ORDER BY rank",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++;
    out_text("fts5rank prime rows="); out_uint((unsigned)n); out_text("\n");
    sqlite3_finalize(st); st=0;
  } else { out_text("fts5rank prime prepare err ("); out_text(sqlite3_errmsg(db1)); out_text(")\n"); }

  /* db2 redefines rank -> %_config row changes, schema cookie bumps. db1's
   * in-memory Fts5Config is now stale but still holds the OLD zRank strings. */
  rc=sqlite3_exec(db2,"INSERT INTO ft(ft, rank) VALUES('rank','bm25(2.0,1.0)');",0,0,&e);
  out_text("fts5rank db2 setrank rc="); out_uint((unsigned)(rc<0?-rc:rc));
  if(e){ out_text(" ("); out_text(e); out_text(")"); sqlite3_free(e); e=0; }
  out_text("\n");

  /* db1 queries again: FilterMethod parses rank off the not-yet-reloaded config
   * (aliasing pConfig->zRank), then the index read triggers ConfigLoad which
   * frees those very strings -> pCsr->zRank dangles and is read again. */
  out_text("fts5rank before stale-rank query\n");
  if(sqlite3_prepare_v2(db1,"SELECT a FROM ft WHERE ft MATCH 'gamma' ORDER BY rank",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++;
    out_text("fts5rank stale rows="); out_uint((unsigned)n); out_text("\n");
    sqlite3_finalize(st); st=0;
  } else { out_text("fts5rank stale prepare err ("); out_text(sqlite3_errmsg(db1)); out_text(")\n"); }

  sqlite3_close(db1); sqlite3_close(db2);
  out_text("fts5rank NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts5rank")
