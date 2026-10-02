/* sqlite_blobclose_domain.c -- CONTROL (unprotected) reproduction of upstream
 * SQLite fix e464802d49: sqlite3_blob_close() uses a lookaside-allocated Incrblob
 * after sqlite3_close_v2() has freed its connection and the lookaside pool.
 * Public, already-fixed bug (fixed in 3.29.0); collected and reproduced for the
 * Capstone/Sublet temporal-safety study.
 *
 * CONTROL arm: SQLite's own memsys5 heap, lookaside ON, nothing revoked. On
 * unprotected Capstone the post-free access is NOT caught, so the domain RETURNS;
 * the matching host ASan build is what shows the heap-use-after-free.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"

#define CAPSTONE_DPI_REGION_SHARE 1U

#ifndef SQLITE_HEAP_SIZE
#define SQLITE_HEAP_SIZE (1024U * 1024U)
#endif
static unsigned char sqlite_heap[SQLITE_HEAP_SIZE] __attribute__((aligned(16)));

static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

static void output_text(const char *text) {
  if (!hostcall_metadata || !hostcall_payload)
    return;
  /* Under the gp-captable (silicon) ABI both capabilities arrive NON-LINEAR (string literals from
     cap-table storage, the payload through the cap-table too), and the RTL's DELIN raises
     UNEXPECTED_CAPABILITY_TYPE on any non-linear operand, a wedge on this RTL, where QEMU's
     helper returns early. Same fix as output_text in sqlite_capstone_domain.c (ISSUES S-02,
     S-15). The QEMU corpus build defines no CAPSTONE_GP_CAPTABLE_ABI and is unchanged. */
#ifdef CAPSTONE_GP_CAPTABLE_ABI
  const char *src = text;
  char *payload = (char *)hostcall_payload;
#else
  const char *src = (const char *)__builtin_capstone_cap_delin((void *)text);
  char *payload = (char *)__builtin_capstone_cap_delin((void *)hostcall_payload);
#endif
  unsigned long offset = hostcall_metadata->length;
  while (*src && offset + 1 < SQLITE_HC_REGION_SIZE)
    payload[offset++] = *src++;
  hostcall_metadata->length = offset;
}

static void output_uint(unsigned long v) {
  char buf[21];
  unsigned i = 21;
  buf[--i] = '\0';
  if (v == 0)
    buf[--i] = '0';
  while (v && i) {
    buf[--i] = (char)('0' + (v % 10));
    v /= 10;
  }
  output_text(&buf[i]);
}

static int fail(const char *stage, int rc) {
  output_text("blobclose SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=");
  output_uint((unsigned long)(rc < 0 ? -rc : rc));
  output_text("\n");
  return rc ? rc : 1;
}

static int run_blobclose(void) {
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
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap,
                          (int)sizeof(sqlite_heap), 64);
  if (rc != SQLITE_OK)
    return fail("config-heap", rc);
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK)
    return fail("initialize", rc);

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
  output_text("blobclose blob_open rc=0\n");

  /* VERIFY the allocator chain instead of assuming it. With lookaside compiled off the
   * high-water mark stays 0; with a pool configured it is the bytes served from slots. */
  {
    int cur = 0, hi = 0;
    int srv = sqlite3_db_status(db, SQLITE_DBSTATUS_LOOKASIDE_USED, &cur, &hi, 0);
    output_text("blobclose lookaside status_rc=");
    output_uint((unsigned long)(srv < 0 ? -srv : srv));
    output_text(" slots_in_use=");
    output_uint((unsigned long)(cur < 0 ? 0 : cur));
    output_text(" high_water=");
    output_uint((unsigned long)(hi < 0 ? 0 : hi));
    /* hi > 0 proves SOME connection-owned allocation was served from a slot; it does not
     * single out the Incrblob. That distinction matters -- see case_lookaside_tagmap.c. */
    output_text(hi > 0 ? "  (lookaside ACTIVE: connection-owned allocations came from slots)\n"
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
  output_text("blobclose close_v2 rc=0 (zombie path taken; plain close would give 5)\n");

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
  output_text("blobclose blob_close rc=0 (the freed-db read already happened)\n");

  output_text("blobclose NOTRAP done\n");
  return 0;
}

void domain_main(unsigned *res, unsigned func) {
  if (func == CAPSTONE_DPI_REGION_SHARE) {
    if (shared_region_count == 0)
      hostcall_metadata = (volatile struct sqlite_hostcall_v0 *)res;
    else if (shared_region_count == 1)
      hostcall_payload = (volatile char *)res;
    ++shared_region_count;
    return;
  }

  if (hostcall_metadata)
    hostcall_metadata->length = 0;

  (void)run_blobclose();

  if (res)
    *res = SQLITE_HC_RET_DONE;
}
