/* fz11_r2 -- R2 fuzz-corpus diff image, trigger re-derived 2026-10-03.
 *
 * Host ASan on stock 3.22.0 (x86): statDecodePage:177925, heap-buffer-overflow
 *
 * The SQL was never recorded for these entries -- fuzzcheck pairs every
 * database with every script and the provenance kept only the database id --
 * so it was re-derived by replaying candidates one process at a time against a
 * fresh copy of the image. It reproduces at the SAME source line as the log.
 *
 * CheriBSD has a real VFS, so the image is opened as an ordinary file; the
 * Capstone arm embeds the same bytes and serves them through repro322_memfs.c.
 * The trigger is identical, so the arms differ by platform alone. */
#include "../repro322_common.h"

#define DBPATH "/root/corpus/fz11_r2.db"

static int run_case(void){
  sqlite3 *db = 0; char *e = 0; int rc;
  if (repro_init()) return 1;

  rc = sqlite3_open(DBPATH, &db);
  out_text("fz11_r2 open_rc="); out_uint((unsigned long)(rc<0?-rc:rc)); out_text("\n");
  if (rc != SQLITE_OK) {
    out_text("fz11_r2 could not open the image; nothing established\n");
    sqlite3_close(db); return 0;
  }

  /* REACHABILITY PROBE. SQLITE_CORRUPT here means SQLite rejected the image
   * before reaching the defect -- a real negative, not a silent pass. */
  rc = sqlite3_exec(db, "PRAGMA integrity_check;", 0, 0, &e);
  out_text("fz11_r2 s0_rc="); out_uint((unsigned long)(rc<0?-rc:rc)); out_text("\n");
  if(e){ out_text("fz11_r2 s0_err: "); out_text(e); out_text("\n"); sqlite3_free(e); e = 0; }

  rc = sqlite3_exec(db, "SELECT * FROM dbstat;", 0, 0, &e);
  out_text("fz11_r2 s1_rc="); out_uint((unsigned long)(rc<0?-rc:rc)); out_text("\n");
  if(e){ out_text("fz11_r2 s1_err: "); out_text(e); out_text("\n"); sqlite3_free(e); e = 0; }

  sqlite3_close(db);
  out_text("fz11_r2 NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("fz11_r2")
