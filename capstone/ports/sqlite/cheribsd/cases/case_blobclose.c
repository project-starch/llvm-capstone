/* case_blobclose.c -- CheriBSD port of the row-1 domain (upstream fix e464802d49:
 * sqlite3_blob_close() reads db->lookaside after sqlite3_close_v2() freed the
 * connection; fixed in 3.30.0). One of the 19 established bugs.
 *
 * Mechanical port of the Capstone original, which predates the group harness and used the
 * old hostcall interface. Only scaffolding changed: out_text/out_uint -> out_text/
 * out_uint, the private sqlite_heap[] plus its sqlite3_config/initialize pair -> repro_init()
 * (same memsys5 arena), domain_main -> REPRO322_MAIN. Trigger and probes untouched.
 */
#include "repro322_common.h"

static int fail(const char *stage, int rc) {
  out_text("blobclose SQLITE ERROR stage="); out_text(stage);
  out_text(" rc="); out_uint((unsigned long)(rc < 0 ? -rc : rc)); out_text("\n");
  return rc ? rc : 1;
}

static int run_case(void) {
  /* memsys5 as the level-0 heap.
   *
   * CORRECTION. An earlier version of this comment said "lookaside stays ON by default, so
   * the Incrblob lands in the connection's lookaside pool". That is wrong for this corpus:
   * build-sqlite-row322.sh sets -DSQLITE_DEFAULT_LOOKASIDE=0,0, and with lookaside off
   * sqlite3DbMallocZero falls straight through to sqlite3Malloc, i.e. memsys5. So in the
   * default configuration this bug exercises ONE allocator, not the nested pair.
   *
   * The bug reproduces either way -- what dangles is the sqlite3 connection itself (a
   * plain sqlite3MallocZero), and the dangling read is of its db->lookaside descriptor
   * field, which sqlite3DbFree consults whether or not a pool exists. Host ASan confirms
   * it fires with lookaside on AND off.
   *
   * To exercise the lookaside > memsys5 chain the paper describes, build with
   * SQLITE_LOOKASIDE=1200,40 (build-sqlite-row322.sh appends that as a later -D, which
   * wins over the 0,0 above). The probe below reports the lookaside high-water mark so the
   * configuration is VERIFIED at runtime rather than assumed. */
  int rc = repro_init();
  if (rc) return rc;

  sqlite3 *db = 0;
  sqlite3_blob *blob = 0;

  /* ==========================================================================
   * row7 style: after each call check rc and `return fail("<stage>", rc);`.
   *
   *   1) open an in-memory db                     -> &db
   *   2) exec: CREATE TABLE t(x);
   *            INSERT INTO t VALUES(zeroblob(16)); (a real blob row to open)
   *   3) blob_open on main.t.x, rowid 1, flags 0  -> &blob (Incrblob -> lookaside)
   *   4) close_v2(db)                             -> zombie close, pool alive
   *   5) blob_close(blob)                         -> 3.22.0 buggy order = the UAF
   * ==========================================================================
   */
  rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK)
    return fail("open", rc);

  rc = sqlite3_exec(db, "CREATE TABLE t(x);"
		  "INSERT INTO t VALUES(zeroblob(16));",
		  0, 0, 0);
  if (rc != SQLITE_OK)
    return fail("exec-setup", rc);

  // Incrblob object lands in the connection lookaside pool
  rc = sqlite3_blob_open(db, "main", "t", "x", 1, 0, &blob);
  if (rc != SQLITE_OK)
    return fail("blob-open", rc);
  out_text("blobclose blob_open rc=0\n");

  /* VERIFY the allocator chain instead of assuming it. With lookaside compiled off the
   * high-water mark stays 0; with a pool configured it is the bytes served from slots. */
  {
    int cur = 0, hi = 0;
    int srv = sqlite3_db_status(db, SQLITE_DBSTATUS_LOOKASIDE_USED, &cur, &hi, 0);
    out_text("blobclose lookaside status_rc=");
    out_uint((unsigned long)(srv < 0 ? -srv : srv));
    out_text(" slots_in_use=");
    out_uint((unsigned long)(cur < 0 ? 0 : cur));
    out_text(" high_water=");
    out_uint((unsigned long)(hi < 0 ? 0 : hi));
    /* hi > 0 proves SOME connection-owned allocation was served from a slot; it does not
     * single out the Incrblob. That distinction matters -- see case_lookaside_tagmap.c. */
    out_text(hi > 0 ? "  (lookaside ACTIVE: connection-owned allocations came from slots)\n"
                       : "  (lookaside OFF: every allocation came straight from memsys5)\n");
  }

  /* blob still open -> connection becomes a zombie, lookaside pool stays alive.
   *
   * REACHABILITY PROBE. close_v2 returning 0 here is the decisive marker: it means the
   * ZOMBIE path was taken, which is what defers the real free of the connection to the
   * finalize inside blob_close below. Plain sqlite3_close() on the same sequence returns
   * SQLITE_BUSY (5) and does not close anything at all, and then there is no UAF --
   * measured on the host, where close_v2 faults under ASan and close is clean. So a 5
   * here, or any nonzero, means this case established nothing. */
  rc = sqlite3_close_v2(db);
  if (rc != SQLITE_OK)
    return fail("close-v2", rc);
  out_text("blobclose close_v2 rc=0 (zombie path taken; plain close would give 5)\n");

  /* 3.22.0's order inside sqlite3_blob_close is
   *     rc = sqlite3_finalize(p->pStmt);   <- last stmt on a zombie db, so
   *                                           sqlite3LeaveMutexAndCloseZombie frees db
   *     sqlite3DbFree(db, p);              <- reads db->lookaside on the FREED db
   * Fixed 3.30.0 by freeing p first and finalizing afterwards.
   *
   * Host ASan oracle on 3.22.0, this exact sequence:
   *   READ of size 8
   *     use   sqlite3DbFreeNN <- sqlite3DbFree <- sqlite3_blob_close
   *     free  sqlite3LeaveMutexAndCloseZombie <- sqlite3_finalize <- sqlite3_blob_close
   * It fires with lookaside ON and with lookaside OFF -- sqlite3DbFree reads
   * db->lookaside.pStart/pEnd either way to decide whether p is a lookaside slot, so
   * this bug is NOT out of a native oracle's reach as was previously recorded. */
  rc = sqlite3_blob_close(blob);
  if (rc != SQLITE_OK)
    return fail("blob-close", rc);
  out_text("blobclose blob_close rc=0 (the freed-db read already happened)\n");

  out_text("blobclose NOTRAP done\n");
  return 0;
}

REPRO322_MAIN("blobclose")
