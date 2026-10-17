/* row15 / sqlite-b9e0f62c3a -- sqlite3_backup caches raw Btree* for src/dest
 * (backup.c:23/29, resolved by findBtree at 82) and re-uses them on every call.
 * DETACHing the destination database closes that Btree (attach.c detachFunc ->
 * sqlite3BtreeClose) while the backup object still holds the pointer, so the next
 * use of the backup reads freed memory. Host ASan on stock 3.22.0:
 *
 *     heap-use-after-free in sqlite3BtreeRollback <- sqlite3_backup_finish
 *     freed by sqlite3BtreeClose <- detachFunc, allocated by sqlite3BtreeOpen <- attachFunc
 *
 * Fixed 3.53.3: both cached pointers become db indices, re-derived at each use
 * (getSourceBtree/getDestBtree), so a detached database yields NULL rather than a
 * stale pointer.
 *
 * WHY THE DETACH IS ALLOWED AT ALL. detachFunc refuses a database that is in a read
 * transaction or in a backup. The second check is sqlite3BtreeIsInBackup, which reads
 * Btree.nBackup -- and sqlite3_backup_init only does pSrc->nBackup++ (backup.c:205).
 * The DESTINATION is never registered. The first check is why the backup is stepped to
 * SQLITE_DONE first: that commits the destination's write transaction (backup.c:567),
 * leaving it in no transaction, while p->bDestLocked stays 1 so no later step reopens one.
 *
 * CONTROL arm: the same sequence with the DETACH removed completes silently, so what the
 * mechanism reports is the detached Btree and not the backup traffic around it.
 */
#include "repro322_common.h"

static int run_case(void) {
  if (repro_init()) return 1;
  sqlite3 *src = 0, *dst = 0;
  char *e = 0;
  int rc = sqlite3_open(":memory:", &src);
  if (rc != SQLITE_OK) return FAILRC("open-src", rc);
  rc = sqlite3_open(":memory:", &dst);
  if (rc != SQLITE_OK) return FAILRC("open-dst", rc);

  rc = sqlite3_exec(src, "CREATE TABLE t(x);INSERT INTO t VALUES(1),(2),(3);", 0,0,0);
  if (rc != SQLITE_OK) return FAILRC("setup-src", rc);

  /* The backup writes into an ATTACHed database of dst, not into its main: only an
   * attached one can be detached (detachFunc refuses i<2). */
  rc = sqlite3_exec(dst, "ATTACH ':memory:' AS aux;", 0, 0, &e);
  out_text("backupattach attach rc="); out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
  if (rc != SQLITE_OK) return FAILRC("attach", rc);

  /* backup object caches src/dst Btree* now (findBtree). */
  sqlite3_backup *bk = sqlite3_backup_init(dst, "aux", src, "main");
  if (!bk) { out_text("backup init failed: "); out_text(sqlite3_errmsg(dst)); out_text("\n");
             return FAILRC("backup-init", SQLITE_ERROR); }

  /* To SQLITE_DONE, which commits the destination's write transaction. */
  do { rc = sqlite3_backup_step(bk, 1); } while (rc == SQLITE_OK);
  out_text("backupattach step-to-done rc="); out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");

  /* REACHABILITY: rc=0 here is the decisive marker -- the DETACH was allowed and the
   * destination Btree is now freed. rc=1 ("database aux is locked") means detachFunc
   * refused it and nothing is freed, which is a real negative, not a silent pass. */
  rc = sqlite3_exec(dst, "DETACH aux;", 0, 0, &e);
  out_text("backupattach detach rc="); out_uint((unsigned)(rc<0?-rc:rc));
  if (e) { out_text(" ("); out_text(e); out_text(")"); sqlite3_free(e); e = 0; }
  out_text("\n");

  /* Uses the cached, now freed, destination Btree. */
  sqlite3_backup_finish(bk);

  sqlite3_close(dst); sqlite3_close(src);
  out_text("backupattach NOTRAP done\n");
  return 0;
}

REPRO322_MAIN("backupattach")
