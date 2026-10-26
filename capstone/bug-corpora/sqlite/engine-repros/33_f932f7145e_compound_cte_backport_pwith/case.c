/* backport / sqlite-f932f7145e -- a real, released SQLite use-after-free reintroduced into 3.22.0
 * by reverting its one-line upstream fix (ports/sqlite/sqlite-pwith-backport.py).
 *
 * convertCompoundSelectToSubquery() rewrites a compound SELECT that carries an ORDER BY COLLATE into
 * a subquery. It shallow-copies the Select (*pNew = *p), so pNew and p momentarily share every
 * pointer field, then nulls each shared field on the original so the two do not both own it. Upstream
 * fix f932f7145e (2015-04-12) added the missing "p->pWith = 0;" for the CTE list; the backport removes
 * it again. With it gone, p and pNew both own the same With object, which is freed on parse teardown
 * (sqlite3SelectDelete -> clearSelect -> sqlite3WithDelete) and then read again through the other
 * Select: a use-after-free (host ASan: heap-use-after-free in sqlite3WithDelete).
 *
 * The With object is served from lookaside: sqlite3DbMallocZero(db, ...) with a live connection,
 * ~48 bytes for one CTE, which SQLITE_WITHIN(db->lookaside.pStart, pEnd) confirms is a slot
 * (native proof, lookaside on: the pointer lies inside the arena). memsys5-level protection
 * (PoisonCap) does not see a slot reuse; a lookaside-aware port (Sublet) does.
 *
 * The bug shipped in released SQLite 3.8.4 through 3.8.8. 3.22.0's function is identical and carries
 * the fix, so this is a BACKPORT, not a native 3.22.0 defect, and is reported on its own line.
 *
 * Trigger is upstream's own regression test (test/with1.test 14.1): a CTE, a compound EXCEPT, and an
 * ORDER BY with a COLLATE term so the rewrite fires. Core group, lookaside on. No string literals, so
 * -DSQLITE_DQS=0 is irrelevant.
 */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  char *e=0;
  rc=sqlite3_exec(db,"CREATE TABLE t(a,b);",0,0,&e);
  if(rc){out_text("pwith setup rc=");out_uint((unsigned)rc);out_text(" (");out_text(e?e:"");out_text(")\n");sqlite3_free(e);e=0;}
  out_text("pwith before trigger\n");
  /* The compound+CTE+ORDER BY COLLATE statement: the rewrite shallow-copies the Select and, with the
   * fix reverted, leaves pNew->pWith and p->pWith aliasing one lookaside slot. */
  rc=sqlite3_exec(db,
    "WITH x AS (SELECT * FROM t) SELECT 0 EXCEPT SELECT 0 ORDER BY 1 COLLATE binary;",0,0,&e);
  out_text("pwith trigger rc=");out_uint((unsigned)rc);out_text(" (");out_text(e?e:"");out_text(")\n");sqlite3_free(e);e=0;
  sqlite3_close(db);
  out_text("pwith NOTRAP done\n"); return 0;
}
REPRO322_MAIN("pwithbackport")
