/* a783931794_0 -- upstream corrupt-database regression image, backported to 3.22.0.
 *
 * Source: the test case added by upstream fix a783931794, whose hex dump was decoded
 * back into a real database file. The defect is present in 3.22.0 (verified: the fix's
 * removed code, or the guarded function without its guard, is in the 3.22.0 tree).
 *
 * Host ASan on stock 3.22.0 crashes on this image; see
 * task-checkpoints/LADYBUG_FUZZDIFF_HAUL.md for the per-image signature.
 *
 * CheriBSD has a real VFS, so the image is opened as an ordinary file. The Capstone arm
 * cannot run this case: its domain is freestanding and the in-memory VFS does not
 * reproduce it (no rollback journal). That asymmetry is deliberate and recorded. */
#include "../repro322_common.h"

#define DBPATH "/root/corpus/a783931794_0.db"

static int run_case(void){
  sqlite3 *db = 0; char *e = 0; int rc;
  if (repro_init()) return 1;

  rc = sqlite3_open(DBPATH, &db);
  out_text("a783931794_0 open_rc="); out_uint((unsigned long)(rc<0?-rc:rc)); out_text("\n");
  if (rc != SQLITE_OK) {
    out_text("a783931794_0 could not open the image; nothing established\n");
    sqlite3_close(db); return 0;
  }

  /* REACHABILITY PROBE. SQLITE_CORRUPT here means SQLite rejected the image before
   * reaching the defect -- a real negative, not a silent pass. */
  rc = sqlite3_exec(db, "PRAGMA integrity_check;", 0, 0, &e);
  out_text("a783931794_0 s0_rc="); out_uint((unsigned long)(rc<0?-rc:rc)); out_text("\n");
  if(e){ out_text("a783931794_0 s0_err: "); out_text(e); out_text("\n"); sqlite3_free(e); e = 0; }
  rc = sqlite3_exec(db, "SELECT count(*) FROM t1 WHERE t1 MATCH '\"json1 enable\"';", 0, 0, &e);
  out_text("a783931794_0 s1_rc="); out_uint((unsigned long)(rc<0?-rc:rc)); out_text("\n");
  if(e){ out_text("a783931794_0 s1_err: "); out_text(e); out_text("\n"); sqlite3_free(e); e = 0; }
  sqlite3_close(db);
  out_text("a783931794_0 NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("a783931794_0")
