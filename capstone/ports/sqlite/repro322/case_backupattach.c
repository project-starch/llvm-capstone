/* row15 / sqlite-b9e0f62c3a -- sqlite3_backup caches raw Btree* for src/dest
 * (backup.c:23/29, resolved by findBtree at 82) and uses them on every step,
 * so the backup outlives any Btree that is closed meanwhile. Fixed 3.53.3
 * (store db indices, re-derive the Btree each step).
 *
 * THE TRIGGER WAS WRONG UNTIL 2026-10-26 AND THIS CASE MEASURED NOTHING.
 * It did a single ATTACH and called that the lifetime ender. Two things are
 * wrong with that. ATTACH frees nothing on the first call: at 3.22.0's
 * attach.c the branch taken when db->aDb is still db->aDbStatic -- which it is
 * on a fresh connection -- is sqlite3DbMallocRawNN plus a memcpy, and only a
 * SECOND attach reaches the sqlite3DbRealloc that frees. And reallocating
 * db->aDb would not matter anyway: what the backup caches is Btree*, the
 * object, not a slot in that array. Nothing was ever freed, so the arms were
 * scoring a NULL dereference -- pBt->pPage1 is 0 in sqlite3BtreeUpdateMeta --
 * which host ASan reported as SEGV on the zero page, the corpus's only SEGV.
 *
 * What does free a Btree is sqlite3BtreeClose, and only DETACH calls it
 * (attach.c detachFunc). detachFunc guards with sqlite3BtreeIsInReadTrans and
 * sqlite3BtreeIsInBackup, and the second one is the hole: it reads nBackup,
 * which backup.c increments on p->pSrc ONLY, so a backup's DESTINATION is
 * unguarded. The first guard still has to be stepped around, which is why
 * nothing may step the backup before the DETACH.
 *
 * CONTROL arm: attach a database, init a backup INTO it, detach it, then step.
 * Host ASan reports heap-use-after-free, an 8-byte read into the freed 72-byte
 * Btree, in sqlite3BtreePager under sqlite3_backup_step, freed by
 * sqlite3BtreeClose under detachFunc.
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

  /* ATTACH first, so there is a DETACHABLE database to back up into. The
   * cached Btree* only dies if some Btree is closed, and only DETACH closes
   * one (attach.c detachFunc -> sqlite3BtreeClose). */
  rc = sqlite3_exec(dst, "ATTACH ':memory:' AS aux;", 0,0,0);
  if (rc != SQLITE_OK) return FAILRC("attach", rc);

  /* backup object caches src/dst Btree* now (findBtree); dest is aux's. */
  sqlite3_backup *bk = sqlite3_backup_init(dst, "aux", src, "main");
  if (!bk) { out_text("backup init failed: "); out_text(sqlite3_errmsg(dst)); out_text("\n");
             return FAILRC("backup-init", SQLITE_ERROR); }

  /* NO step before the DETACH: the first step opens a write transaction on the
   * destination, and detachFunc refuses on sqlite3BtreeIsInReadTrans. init has
   * already cached both Btree* through findBtree, which is all the defect needs. */

  /* DETACH closes aux's Btree and frees it. detachFunc's guard is
   * sqlite3BtreeIsInBackup, which reads nBackup, and backup.c increments
   * nBackup on p->pSrc ONLY -- the DESTINATION is unguarded, so this
   * succeeds while the backup still holds the pointer. */
  rc = sqlite3_exec(dst, "DETACH aux;", 0,0,0);
  if (rc != SQLITE_OK) return FAILRC("detach", rc);

  /* step again: uses the freed cached dest Btree*. */
  rc = sqlite3_backup_step(bk, -1);
  out_text("backup step2 rc="); out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
  sqlite3_backup_finish(bk);

  sqlite3_close(src); sqlite3_close(dst);
  out_text("backupattach NOTRAP done\n");
  return 0;
}

REPRO322_MAIN("backupattach")
