/* fz06_r2 -- R2 fuzz-corpus diff image, trigger re-derived 2026-10-03.
 *
 * Host ASan on stock 3.22.0 (x86): vdbeRecordCompareInt:77204, heap-buffer-overflow
 *
 * The SQL was never recorded for these entries -- fuzzcheck pairs every
 * database with every script and the provenance kept only the database id --
 * so it was re-derived by replaying candidates one process at a time against a
 * fresh copy of the image. It reproduces at the SAME source line as the log.
 *
 * CheriBSD has a real VFS, so the image is opened as an ordinary file; the
 * Capstone arm embeds the same bytes and serves them through repro322_memfs.c.
 * That VFS difference is the ONLY intended difference: the statement that reaches
 * the defect is the same one, in the same position, on both arms (corrected
 * 2026-10-17; see the comment on the trigger below). */
#include "../repro322_common.h"

#define DBPATH "/root/corpus/fz06_r2.db"

static int run_case(void){
  sqlite3 *db = 0; char *e = 0; int rc;
  if (repro_init()) return 1;

  rc = sqlite3_open(DBPATH, &db);
  out_text("fz06_r2 open_rc="); out_uint((unsigned long)(rc<0?-rc:rc)); out_text("\n");
  if (rc != SQLITE_OK) {
    out_text("fz06_r2 could not open the image; nothing established\n");
    sqlite3_close(db); return 0;
  }

  /* The trigger, and it runs FIRST -- the same statement the corpus copy runs as s0.
   * It used to run after a PRAGMA integrity_check that was meant as a reachability
   * probe: on a corrupt image that pragma makes SQLite reject this SELECT with
   * SQLITE_CORRUPT, so the defect was never reached on this arm and the run scored
   * as a clean completion. SQLITE_CORRUPT HERE is a real negative; after the pragma
   * it told us nothing about the defect. */
  rc = sqlite3_exec(db, "SELECT * FROM t3 WHERE x > 5;", 0, 0, &e);
  out_text("fz06_r2 s0_rc="); out_uint((unsigned long)(rc<0?-rc:rc)); out_text("\n");
  if(e){ out_text("fz06_r2 s0_err: "); out_text(e); out_text("\n"); sqlite3_free(e); e = 0; }

  /* Kept for what it reports about the image, now where it cannot gate the defect. */
  rc = sqlite3_exec(db, "PRAGMA integrity_check;", 0, 0, &e);
  out_text("fz06_r2 s1_rc="); out_uint((unsigned long)(rc<0?-rc:rc)); out_text("\n");
  if(e){ out_text("fz06_r2 s1_err: "); out_text(e); out_text("\n"); sqlite3_free(e); e = 0; }

  sqlite3_close(db);
  out_text("fz06_r2 NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("fz06_r2")
