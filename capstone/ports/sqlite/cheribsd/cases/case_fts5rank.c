/* row 12 / sqlite-12439f9c16 -- fts5 reads a freed Fts5Config rank string.
 *
 * fts5FilterMethod() parses rank from a config that has NOT yet been reloaded, and
 * fts5CursorParseRank() ALIASES the config strings rather than copying them:
 *     pCsr->zRank     = (char*)pConfig->zRank;
 *     pCsr->zRankArgs = (char*)pConfig->zRankArgs;
 * When another connection redefines rank, %_config changes and the schema cookie bumps;
 * the later sqlite3Fts5ConfigLoad -> sqlite3Fts5ConfigSetValue(..."rank"...) frees those
 * very strings, so the cursor's aliases dangle and are read again. Fixed 3.43.2 by
 * hoisting sqlite3Fts5IndexLoadConfig() above the rank parsing.
 *
 * WHAT CHANGED HERE, AND WHY. An earlier version of this case queried only column `a`
 * and never read `rank`. That reached the stale config but never called the rank
 * function on it, so the dangling string was never dereferenced -- a PASS that
 * established nothing. The use site is fts5FindRankFunction() formatting zRankArgs
 * through Mprintf, which only runs when the rank VALUE is read. The query now selects
 * `rank<0`, as upstream's own test does.
 *
 * Host ASan oracle (3.22.0), identical in upstream's two-connections-on-one-file
 * arrangement AND in the shared-cache arrangement this domain must use:
 *   READ of size 3 (the zRankArgs text)
 *     use   sqlite3VXPrintf <- sqlite3_vmprintf <- sqlite3Fts5Mprintf
 *             <- fts5FindRankFunction <- fts5ColumnMethod
 *     free  sqlite3Fts5ConfigSetValue <- sqlite3Fts5ConfigLoad
 *             <- fts5StructureReadUncached
 *
 * REACHABILITY PROBE -- a three-point signature that cannot occur unless the stale
 * config was really parsed. Measured on the host, same SQL:
 *     query 1 (fresh config)  : 2 rows, rank<0 true for both
 *     query 2 (stale config)  : 0 rows, rc=1 SQLITE_ERROR     <-- the dangling read
 *     query 3 (reloaded)      : 2 rows again
 * Upstream's fixed build returns 2 rows for query 2, so the error IS the bug showing.
 *
 * WHY THE WEIGHTS ARE INTEGERS. fts5 parses the configured rank string as SQL, and this
 * port compiles with -DSQLITE_OMIT_FLOATING_POINT=1, which puts the float-literal branch
 * of sqlite3GetToken() behind an #ifndef. Under that flag '10.0' tokenizes as TK_INTEGER
 * "10" followed by a stray '.', so upstream's bm25(10.0,1.0) fails to parse and EVERY
 * query that reads rank returns `near ".": syntax error`. That is exactly how this case
 * first came back with 0 rows on all three queries -- the reachability probe caught it.
 * bm25(10,1) parses, and the host oracle confirms the integer weights reproduce the same
 * use-after-free at the same site.
 *
 * Needs TWO connections on ONE database. The base build sets
 * -DSQLITE_OMIT_SHARED_CACHE=1, so this builds in the fts5S group which undefines it;
 * a :memory: database, shared-cache URI form included, takes SQLite's memDb path and
 * never calls the VFS xOpen (which always returns SQLITE_CANTOPEN here).
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

#define URI "file::memory:?cache=shared"

static void rcline(const char *what, int rc){
  out_text("fts5rank "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

static int openshared(sqlite3 **pdb, const char *tag){
  int rc = sqlite3_open_v2(URI, pdb,
      SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|SQLITE_OPEN_URI, 0);
  out_text("fts5rank open "); out_text(tag); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc));
  if(rc!=SQLITE_OK && *pdb){ out_text(" ("); out_text(sqlite3_errmsg(*pdb)); out_text(")"); }
  out_text("\n");
  return rc;
}

/* SELECT a, b, rank<0 FROM t WHERE t MATCH 'data*' ORDER BY rank
 * Reading the rank column is what drives fts5FindRankFunction over zRankArgs. */
static int rankq(sqlite3 *db, const char *label, int *pNeg, int *pEndRc){
  const char *sql = "SELECT a, b, rank<0 FROM t WHERE t MATCH 'data*' ORDER BY rank";
  sqlite3_stmt *st = 0;
  int n = 0, neg = 0;
  *pNeg = 0; *pEndRc = -1;
  int rc = sqlite3_prepare_v2(db, sql, -1, &st, 0);
  if(rc!=SQLITE_OK){
    out_text("fts5rank "); out_text(label); out_text(" prepare rc=");
    out_uint((unsigned)(rc<0?-rc:rc));
    out_text(" ("); out_text(sqlite3_errmsg(db)); out_text(")\n");
    *pEndRc = rc;
    return 0;
  }
  while(sqlite3_step(st)==SQLITE_ROW){ n++; if(sqlite3_column_int(st,2)) neg++; }
  *pEndRc = sqlite3_errcode(db);
  if(*pEndRc!=SQLITE_OK && *pEndRc!=SQLITE_DONE){
    out_text("fts5rank "); out_text(label); out_text(" step err (");
    out_text(sqlite3_errmsg(db)); out_text(")\n");
  }
  sqlite3_finalize(st);
  out_text("fts5rank "); out_text(label); out_text(" rows=");
  out_uint((unsigned)n);
  out_text(" rank_negative="); out_uint((unsigned)neg);
  out_text(" end_rc="); out_uint((unsigned)(*pEndRc<0?-*pEndRc:*pEndRc));
  out_text("\n");
  *pNeg = neg;
  return n;
}

static int run_case(void){
  if (repro_init()) return 1;
  char *e=0; int rc;

  rc = sqlite3_enable_shared_cache(1);
  rcline("shared_cache", rc);

  sqlite3 *db1=0, *db2=0;
  if(openshared(&db1,"db1")){ sqlite3_close(db1); return 1; }

  rc = sqlite3_exec(db1,
      "CREATE VIRTUAL TABLE t USING fts5(a, b);"
      "INSERT INTO t (a, b) VALUES ('data1','sentence1'), ('data2','sentence2');", 0,0,&e);
  rcline("setup", rc);
  if(e){ out_text("fts5rank setup err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db1); return 1; }

  rc = sqlite3_exec(db1, "INSERT INTO t(t, rank) VALUES ('rank','bm25(10,1)');",0,0,&e);
  rcline("db1 setrank", rc);
  if(e){ out_text("fts5rank setrank err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }

  /* db2 is a second connection on the same shared-cache database. */
  if(openshared(&db2,"db2")){ sqlite3_close(db1); sqlite3_close(db2); return 1; }

  int n1,n2,n3, g1,g2,g3, r1,r2,r3;
  out_text("fts5rank -- query 1: fresh config, cursor aliases zRank/zRankArgs\n");
  n1 = rankq(db2,"q1 fresh",&g1,&r1);

  /* DECISIVE that the two connections really share one fts5 table: if they did not,
   * this would fail with "no such table: t". */
  out_text("fts5rank -- db1 redefines rank: %_config changes, cookie bumps\n");
  rc = sqlite3_exec(db1, "INSERT INTO t(t, rank) VALUES ('rank','bm25(10,1)');",0,0,&e);
  rcline("db1 setrank again", rc);
  if(e){ out_text("fts5rank setrank2 err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  int shared_ok = (rc==SQLITE_OK);

  out_text("fts5rank -- query 2: parses rank off the STALE config, then the reload frees it\n");
  n2 = rankq(db2,"q2 stale",&g2,&r2);

  out_text("fts5rank -- query 3: config reloaded, should recover\n");
  n3 = rankq(db2,"q3 reloaded",&g3,&r3);

  /* The signature: 2 rows / error / 2 rows. */
  out_text("fts5rank VERDICT shared_ok="); out_uint((unsigned)shared_ok);
  out_text(" q1_rows="); out_uint((unsigned)n1);
  out_text(" q2_rows="); out_uint((unsigned)n2);
  out_text(" q2_err="); out_uint((unsigned)(r2!=SQLITE_OK && r2!=SQLITE_DONE));
  out_text(" q3_rows="); out_uint((unsigned)n3);
  out_text(" stale_read_reached=");
  out_uint((unsigned)(shared_ok && n1==2 && n2==0 && n3==2));
  out_text("\n");
  if(!shared_ok)
    out_text("fts5rank WARNING db1 could not redefine rank; the two connections did"
             " not share one table, so nothing was staled\n");
  else if(!(n1==2 && n2==0 && n3==2))
    out_text("fts5rank WARNING the 2/error/2 signature did not hold; the stale-config"
             " rank parse may not have been reached\n");

  sqlite3_close(db1); sqlite3_close(db2);
  out_text("fts5rank NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts5rank")
