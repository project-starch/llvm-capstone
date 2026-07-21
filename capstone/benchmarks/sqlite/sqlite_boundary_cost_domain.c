/* Tier-2 boundary-cost domain: run a bulk SQLite SELECT scan IN a pure-capability
 * domain and measure, in the SAME environment, the two quantities the host<->engine
 * boundary overhead depends on:
 *
 *   T  = retired instructions of the real in-domain SQLite scan (QEMU -icount);
 *   B  = boundary events = sqlite3_column_text hand-outs (one per text column per row).
 *
 * This replaces Tier-1's native-x86 denominator with the true target-ISA, in-domain
 * instruction count of the SQLite work. The boundary *unit* cost stays the silicon
 * super-op (171 cyc, mrev+delin+load+revoke) -- QEMU -icount cannot see the
 * multi-cycle mrev/revoke HW cost (it counts each as ~1 instruction), so a QEMU
 * cycle number would understate the boundary op by ~30x. Overhead estimate is then
 * B*171 / (T * CPI), same model as Tier-1 but with a target-faithful denominator.
 *
 * Built via build-sqlite-capstone.sh with DOMAIN_SRC=this file. Run under -icount.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"

#define CAPSTONE_DPI_REGION_SHARE 1U
#define SQLITE_HEAP_SIZE (1024U * 1024U)
#ifndef BOUNDARY_ROWS
#define BOUNDARY_ROWS 200U
#endif

static unsigned char sqlite_heap[SQLITE_HEAP_SIZE] __attribute__((aligned(16)));
static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

#define CAPSTONE_DELIN(value)                                                \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(value))

/* csrdicount rd -- QEMU's retired-instruction count (only meaningful under -icount;
 * returns 0 otherwise). Same op the borrow-cost probe uses. */
static inline unsigned long rd_icount(void) {
  unsigned long v;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x48, %0, x0, x0" : "=r"(v) : : "memory");
  return v;
}

static void output_text(const char *text) {
  if (!hostcall_metadata || !hostcall_payload)
    return;
  CAPSTONE_DELIN(text);
  char *payload = (char *)hostcall_payload;
  CAPSTONE_DELIN(payload);
  unsigned long offset = hostcall_metadata->length;
  while (*text && offset + 1 < SQLITE_HC_REGION_SIZE)
    payload[offset++] = *text++;
  hostcall_metadata->length = offset;
}

static void output_ulong(unsigned long value) {
  char digits[24];
  unsigned count = 0;
  do {
    digits[count++] = (char)('0' + value % 10UL);
    value /= 10UL;
  } while (value != 0);
  while (count != 0) {
    char one[2] = {digits[--count], '\0'};
    output_text(one);
  }
}

static int fail(const char *stage, int rc, sqlite3 *db) {
  output_text("SQLITE ERROR stage=");
  output_text(stage);
  output_text(" rc=");
  output_ulong((unsigned long)rc);
  if (db) {
    output_text(" message=");
    output_text(sqlite3_errmsg(db));
  }
  output_text("\n");
  return rc ? rc : 1;
}

static int exec_ok(sqlite3 *db, const char *sql, const char *stage) {
  int rc = sqlite3_exec(db, sql, 0, 0, 0);
  if (rc != SQLITE_OK)
    return fail(stage, rc, db);
  return SQLITE_OK;
}

/* Sink kept live so the optimiser cannot delete the column reads being measured. */
static volatile unsigned long sink;

static int run_boundary_cost(void) {
  sqlite3 *db = 0;
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap,
                          (int)sizeof(sqlite_heap), 64);
  if (rc != SQLITE_OK)
    return fail("config-heap", rc, 0);
  rc = sqlite3_initialize();
  if (rc != SQLITE_OK)
    return fail("initialize", rc, 0);
  rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK)
    return fail("open", rc, db);

  if ((rc = exec_ok(db,
          "CREATE TABLE t(id INTEGER PRIMARY KEY, a TEXT, b TEXT);",
          "create")) != SQLITE_OK)
    return rc;

  /* Bulk insert BOUNDARY_ROWS rows with two TEXT columns (static-bound). */
  if ((rc = exec_ok(db, "BEGIN;", "begin")) != SQLITE_OK)
    return rc;
  {
    sqlite3_stmt *ins = 0;
    rc = sqlite3_prepare_v2(db, "INSERT INTO t(id,a,b) VALUES(?1,?2,?3);", -1,
                            &ins, 0);
    if (rc != SQLITE_OK)
      return fail("prep-insert", rc, db);
    /* A pool of distinct static labels so column_text returns real strings. */
    static const char *pool[8] = {"alpha", "bravo", "charlie", "delta",
                                  "echo",  "foxtrot", "golf",   "hotel"};
    for (unsigned i = 0; i < BOUNDARY_ROWS; ++i) {
      sqlite3_reset(ins);
      sqlite3_bind_int(ins, 1, (int)(i + 1));
      sqlite3_bind_text(ins, 2, pool[i & 7U], -1, SQLITE_STATIC);
      sqlite3_bind_text(ins, 3, pool[(i + 3U) & 7U], -1, SQLITE_STATIC);
      rc = sqlite3_step(ins);
      if (rc != SQLITE_DONE) {
        sqlite3_finalize(ins);
        return fail("bind-insert", rc, db);
      }
    }
    sqlite3_finalize(ins);
  }
  if ((rc = exec_ok(db, "COMMIT;", "commit")) != SQLITE_OK)
    return rc;

  /* The measured scan: SELECT the two TEXT columns from every row, reading each
   * through sqlite3_column_text (the boundary hand-out / borrow site). Bracket the
   * whole scan with rd_icount(); count B = column_text calls. */
  sqlite3_stmt *sel = 0;
  rc = sqlite3_prepare_v2(db, "SELECT a,b FROM t ORDER BY id;", -1, &sel, 0);
  if (rc != SQLITE_OK)
    return fail("prep-select", rc, db);

  unsigned long borrows = 0, rows = 0, acc = 0;
  unsigned long t0 = rd_icount();
  while ((rc = sqlite3_step(sel)) == SQLITE_ROW) {
    const unsigned char *za = sqlite3_column_text(sel, 0);
    const unsigned char *zb = sqlite3_column_text(sel, 1);
    borrows += 2;
    /* Touch the borrowed bytes so the hand-out is real work, not dead code. */
    if (za) acc += za[0];
    if (zb) acc += zb[0];
    ++rows;
  }
  unsigned long t1 = rd_icount();
  sink += acc;
  sqlite3_finalize(sel);
  if (rc != SQLITE_DONE)
    return fail("scan", rc, db);

  unsigned long T = t1 - t0;

  output_text("__BOUNDARY_COST__ rows=");
  output_ulong(rows);
  output_text(" borrows=");
  output_ulong(borrows);
  output_text(" scan_instrs=");
  output_ulong(T);
  if (borrows) {
    output_text(" instrs_per_borrow=");
    output_ulong(T / borrows);
  }
  output_text("\n");

  rc = sqlite3_close(db);
  if (rc != SQLITE_OK)
    return fail("close", rc, 0);

  output_text("__CAPSTONE_SQLITE_BOUNDARY_COST_PASSED__\n");
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
  (void)run_boundary_cost();
  *res = SQLITE_HC_RET_DONE;
}
