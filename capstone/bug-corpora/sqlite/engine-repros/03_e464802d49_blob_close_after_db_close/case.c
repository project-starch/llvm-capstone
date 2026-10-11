/* row05 / sqlite-e464802d49 -- sqlite3_blob_close() reads the connection after
 * sqlite3_close_v2() has freed it. 3.22.0's order inside sqlite3_blob_close is
 *     rc = sqlite3_finalize(p->pStmt);   <- last stmt on a zombie db, so
 *                                           sqlite3LeaveMutexAndCloseZombie frees db
 *     sqlite3DbFree(db, p);              <- reads db->lookaside on the FREED db
 * Fixed 3.29.0 by freeing p first and finalizing afterwards.
 *
 * Host ASan oracle on 3.22.0, this exact sequence:
 *   READ of size 8
 *     use   sqlite3DbFreeNN <- sqlite3DbFree <- sqlite3_blob_close
 *     free  sqlite3LeaveMutexAndCloseZombie <- sqlite3_finalize <- sqlite3_blob_close
 *
 * CONTROL arm: SQLite's own memsys5 heap, nothing revoked. On unprotected
 * Capstone the post-free access is not caught, so the domain RETURNS.
 *
 * PORTED 2026-10-05 from the legacy standalone domain onto REPRO322_MAIN. The
 * old file carried its own domain_main and its own hostcall plumbing, and never
 * captured the Sublet grant -- so under Sublet memsys5Init read an empty slot and
 * faulted inside capstone_cap_base BEFORE the case ran, and corpus322.sh skipped
 * this tag on that arm entirely. The defect sequence below is unchanged; only the
 * entry point and the output helpers moved to the shared harness.
 */
#include "repro322_common.h"

static int run_case(void) {
  /* memsys5 as the level-0 heap.
   *
   * CORRECTION (carried over from the port copy; the corpus copy of this file
   * still had the superseded claim). An earlier version of this comment said
   * "lookaside stays ON by default, so the Incrblob lands in the connection's
   * lookaside pool". That is wrong for this corpus: build-sqlite-row322.sh sets
   * -DSQLITE_DEFAULT_LOOKASIDE=0,0, and with lookaside off sqlite3DbMallocZero
   * falls straight through to sqlite3Malloc, i.e. memsys5. So in the default
   * configuration this bug exercises ONE allocator, not the nested pair.
   *
   * The bug reproduces either way -- what dangles is the sqlite3 connection
   * itself (a plain sqlite3MallocZero), and the dangling read is of its
   * db->lookaside descriptor field, which sqlite3DbFree consults whether or not
   * a pool exists. Host ASan confirms it fires with lookaside on AND off.
   *
   * To exercise the lookaside > memsys5 chain the paper describes, build with
   * SQLITE_LOOKASIDE=1200,40. The probe below reports the lookaside high-water
   * mark so the configuration is VERIFIED at runtime rather than assumed. */
  if (repro_init()) return 1;

  sqlite3 *db = 0;
  sqlite3_blob *blob = 0;

  int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) return FAILRC("open", rc);

  rc = sqlite3_exec(db, "CREATE TABLE t(x);"
                        "INSERT INTO t VALUES(zeroblob(16));",
                    0, 0, 0);
  if (rc != SQLITE_OK) return FAILRC("exec-setup", rc);

  rc = sqlite3_blob_open(db, "main", "t", "x", 1, 0, &blob);
  if (rc != SQLITE_OK) return FAILRC("blob-open", rc);
  out_text("blobclose blob_open rc=0\n");

  /* VERIFY the allocator chain instead of assuming it. With lookaside compiled
   * off the high-water mark stays 0; with a pool configured it is the bytes
   * served from slots. */
  {
    int cur = 0, hi = 0;
    int srv = sqlite3_db_status(db, SQLITE_DBSTATUS_LOOKASIDE_USED, &cur, &hi, 0);
    out_text("blobclose lookaside status_rc=");
    out_uint((unsigned)(srv < 0 ? -srv : srv));
    out_text(" slots_in_use=");
    out_uint((unsigned)(cur < 0 ? 0 : cur));
    out_text(" high_water=");
    out_uint((unsigned)(hi < 0 ? 0 : hi));
    /* hi > 0 proves SOME connection-owned allocation was served from a slot; it
     * does not single out the Incrblob. */
    out_text(hi > 0 ? "  (lookaside ACTIVE: connection-owned allocations came from slots)\n"
                    : "  (lookaside OFF: every allocation came straight from memsys5)\n");
  }

  /* blob still open -> connection becomes a zombie, the real free deferred to the
   * finalize inside blob_close below.
   *
   * REACHABILITY PROBE. close_v2 returning 0 here is the decisive marker: it
   * means the ZOMBIE path was taken. Plain sqlite3_close() on the same sequence
   * returns SQLITE_BUSY (5) and closes nothing, and then there is no UAF --
   * measured on the host, where close_v2 faults under ASan and close is clean.
   * So a 5 here, or any nonzero, means this case established nothing. */
  rc = sqlite3_close_v2(db);
  if (rc != SQLITE_OK) return FAILRC("close-v2", rc);
  out_text("blobclose close_v2 rc=0 (zombie path taken; plain close would give 5)\n");

  rc = sqlite3_blob_close(blob);
  if (rc != SQLITE_OK) return FAILRC("blob-close", rc);
  out_text("blobclose blob_close rc=0 (the freed-db read already happened)\n");

  out_text("blobclose NOTRAP done\n");
  return 0;
}

REPRO322_MAIN("blobclose")
