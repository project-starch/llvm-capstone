/* row15 / sqlite-b9e0f62c3a -- sqlite3_backup caches raw Btree* for src/dest
 * (backup.c:23/29, resolved by findBtree at 82). ATTACH reallocates db->aDb
 * (attach.c:113 sqlite3DbRealloc) and can invalidate the cached Btree*, so a
 * later backup step uses a freed/moved Btree. Fixed 3.53.3 (store db indices,
 * re-derive the Btree each step).
 *
 * CONTROL arm: start a backup, then ATTACH a database on the SAME connection to
 * force db->aDb realloc, then step the backup -- it dereferences the stale
 * Btree*. On unprotected Capstone the moved-Btree access is not caught, so the
 * step returns and the domain RETURNS. Host ASan flags heap-use-after-free.
 */
#include "repro322_common.h"

static int run_case(void) {
  if (repro_init()) return 1;
  sqlite3 *src = 0, *dst = 0;
  int rc = sqlite3_open(":memory:", &src);
  if (rc != SQLITE_OK) return FAILRC("open-src", rc);
  rc = sqlite3_open(":memory:", &dst);
  if (rc != SQLITE_OK) return FAILRC("open-dst", rc);

  rc = sqlite3_exec(src, "CREATE TABLE t(x);INSERT INTO t VALUES(1),(2),(3);", 0,0,0);
  if (rc != SQLITE_OK) return FAILRC("setup-src", rc);

  /* backup object caches src/dst Btree* now (findBtree). */
  sqlite3_backup *bk = sqlite3_backup_init(dst, "main", src, "main");
  if (!bk) { out_text("backup init failed: "); out_text(sqlite3_errmsg(dst)); out_text("\n");
             return FAILRC("backup-init", SQLITE_ERROR); }

  /* one page, so the backup is mid-operation, not finished */
  rc = sqlite3_backup_step(bk, 1);
  out_text("backup step1 rc="); out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");

  /* ATTACH on dst forces db->aDb realloc -> cached dest Btree* may move. */
  rc = sqlite3_exec(dst, "ATTACH ':memory:' AS aux;", 0,0,0);
  if (rc != SQLITE_OK) return FAILRC("attach", rc);

  /* step again: uses the (possibly moved) cached Btree*. */
  rc = sqlite3_backup_step(bk, -1);
  out_text("backup step2 rc="); out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
  sqlite3_backup_finish(bk);

  sqlite3_close(src); sqlite3_close(dst);
  out_text("backupattach NOTRAP done\n");
  return 0;
}

REPRO322_MAIN("backupattach")
