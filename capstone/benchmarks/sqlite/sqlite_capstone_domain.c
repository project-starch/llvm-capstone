#include "sqlite3.h"
#include "sqlite_hostcall.h"

#define CAPSTONE_DPI_REGION_SHARE 1U
#define SQLITE_HEAP_SIZE (1024U * 1024U)

static unsigned char sqlite_heap[SQLITE_HEAP_SIZE] __attribute__((aligned(16)));
static volatile struct sqlite_hostcall_v0 *hostcall_metadata;
static volatile char *hostcall_payload;
static unsigned shared_region_count;

#define CAPSTONE_DELIN(value)                                                \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(value))

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

static void output_uint(unsigned value) {
  char digits[16];
  unsigned count = 0;
  do {
    digits[count++] = (char)('0' + value % 10U);
    value /= 10U;
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
  output_uint((unsigned)rc);
  if (db) {
    output_text(" message=");
    output_text(sqlite3_errmsg(db));
  }
  output_text("\n");
  return rc ? rc : 1;
}

/* Run a statement with no result rows; fail() on error. */
static int exec_ok(sqlite3 *db, const char *sql, const char *stage) {
  char *errmsg = 0;
  int rc = sqlite3_exec(db, sql, 0, 0, &errmsg);
  if (errmsg)
    sqlite3_free(errmsg);
  if (rc != SQLITE_OK)
    return fail(stage, rc, db);
  return SQLITE_OK;
}

/* Prepare + step a query expected to return exactly one row, one integer
 * column; check it equals `want`. Exercises prepare/step/column + finalize. */
static int query_scalar_eq(sqlite3 *db, const char *sql, long want,
                           const char *stage) {
  sqlite3_stmt *st = 0;
  int rc = sqlite3_prepare_v2(db, sql, -1, &st, 0);
  if (rc != SQLITE_OK)
    return fail(stage, rc, db);
  rc = sqlite3_step(st);
  if (rc != SQLITE_ROW) {
    sqlite3_finalize(st);
    return fail(stage, rc ? rc : SQLITE_MISMATCH, db);
  }
  long got = (long)sqlite3_column_int64(st, 0);
  sqlite3_finalize(st);
  if (got != want)
    return fail(stage, SQLITE_MISMATCH, db);
  return SQLITE_OK;
}

/* A richer workload: exercises transactions, a second table, a REAL column, an
 * INTEGER PRIMARY KEY (rowid btree) + a secondary INDEX (index btree cursors),
 * bound prepared inserts, UPDATE/DELETE, aggregates + the sorter, ORDER BY, a
 * JOIN, GROUP BY, and string functions (sqlite3UpperToLower). Each step is
 * validated; the first failure returns non-zero and the PASSED marker is skipped. */
static int run_sqlite_extended(sqlite3 *db) {
  int rc;

  if ((rc = exec_ok(db, "BEGIN;", "ext-begin")) != SQLITE_OK)
    return rc;
  if ((rc = exec_ok(db,
          "CREATE TABLE nums(id INTEGER PRIMARY KEY, label TEXT, amount REAL);",
          "ext-create-nums")) != SQLITE_OK)
    return rc;
  if ((rc = exec_ok(db, "CREATE INDEX idx_amount ON nums(amount);",
                    "ext-create-index")) != SQLITE_OK)
    return rc;

  /* Bound prepared INSERTs: int id, text label, real amount. */
  {
    sqlite3_stmt *ins = 0;
    rc = sqlite3_prepare_v2(
        db, "INSERT INTO nums(id,label,amount) VALUES(?1,?2,?3);", -1, &ins, 0);
    if (rc != SQLITE_OK)
      return fail("ext-prep-insert", rc, db);
    static const char *labels[4] = {"one", "two", "three", "four"};
    for (int i = 0; i < 4; ++i) {
      sqlite3_reset(ins);
      sqlite3_bind_int(ins, 1, i + 1);
      /* SQLITE_STATIC (not SQLITE_TRANSIENT): these labels are static storage.
       * The build patches SQLITE_TRANSIENT to a function only inside the
       * amalgamation, so the public sqlite3.h value (-1) is not recognized as
       * the transient sentinel by the patched core -- a client passing
       * SQLITE_TRANSIENT would have -1 stored as a destructor and later called.
       * Tracked as a known limitation; SQLITE_STATIC is the correct binding for
       * persistent buffers regardless. */
      sqlite3_bind_text(ins, 2, labels[i], -1, SQLITE_STATIC);
      sqlite3_bind_double(ins, 3, (double)((i + 1) * 10));
      rc = sqlite3_step(ins);
      if (rc != SQLITE_DONE) {
        sqlite3_finalize(ins);
        return fail("ext-bind-insert", rc, db);
      }
    }
    sqlite3_finalize(ins);
  }

  /* Mutations. */
  if ((rc = exec_ok(db, "UPDATE nums SET amount = amount * 2 WHERE id = 2;",
                    "ext-update")) != SQLITE_OK)
    return rc;
  if ((rc = exec_ok(db, "DELETE FROM nums WHERE id = 4;", "ext-delete")) !=
      SQLITE_OK)
    return rc;
  if ((rc = exec_ok(db, "COMMIT;", "ext-commit")) != SQLITE_OK)
    return rc;

  /* Aggregates over the remaining rows: id 1,2,3 with amounts 10,40,30.
   * count=3, sum=80, max=40, min=10. */
  if ((rc = query_scalar_eq(db, "SELECT COUNT(*) FROM nums;", 3,
                            "ext-count")) != SQLITE_OK)
    return rc;
  if ((rc = query_scalar_eq(db, "SELECT CAST(SUM(amount) AS INTEGER) FROM nums;",
                            80, "ext-sum")) != SQLITE_OK)
    return rc;
  if ((rc = query_scalar_eq(db, "SELECT CAST(MAX(amount) AS INTEGER) FROM nums;",
                            40, "ext-max")) != SQLITE_OK)
    return rc;

  /* ORDER BY (sorter): top amount label should be "two" (40). */
  {
    sqlite3_stmt *st = 0;
    rc = sqlite3_prepare_v2(
        db, "SELECT label FROM nums ORDER BY amount DESC;", -1, &st, 0);
    if (rc != SQLITE_OK)
      return fail("ext-prep-order", rc, db);
    rc = sqlite3_step(st);
    if (rc != SQLITE_ROW) {
      sqlite3_finalize(st);
      return fail("ext-order-step", rc, db);
    }
    const unsigned char *top = sqlite3_column_text(st, 0);
    int ok = top && sqlite3_stricmp((const char *)top, "two") == 0;
    sqlite3_finalize(st);
    if (!ok)
      return fail("ext-order-value", SQLITE_MISMATCH, db);
  }

  /* Index-driven lookup (WHERE on the indexed column), comparison ops. */
  if ((rc = query_scalar_eq(
           db, "SELECT id FROM nums WHERE amount > 35 AND amount < 45;", 2,
           "ext-index-where")) != SQLITE_OK)
    return rc;

  /* JOIN nums with the base items table (both TEXT keys). */
  if ((rc = exec_ok(db,
          "CREATE TABLE link(label TEXT, name TEXT);", "ext-create-link")) !=
      SQLITE_OK)
    return rc;
  if ((rc = exec_ok(db,
          "INSERT INTO link VALUES('one','alpha'),('two','beta');",
          "ext-insert-link")) != SQLITE_OK)
    return rc;
  if ((rc = query_scalar_eq(
           db,
           "SELECT COUNT(*) FROM nums JOIN link ON nums.label = link.label;", 2,
           "ext-join")) != SQLITE_OK)
    return rc;

  /* GROUP BY / HAVING: two parity groups over id (1,2,3) -> odd{1,3}, even{2}. */
  if ((rc = query_scalar_eq(
           db,
           "SELECT COUNT(*) FROM (SELECT id%2 AS p FROM nums GROUP BY id%2);", 2,
           "ext-group")) != SQLITE_OK)
    return rc;

  /* String functions: exercises sqlite3UpperToLower / substr / length. */
  if ((rc = query_scalar_eq(db, "SELECT length(upper(label)) FROM nums WHERE id=3;",
                            5, "ext-strfunc")) != SQLITE_OK) /* "THREE" -> 5 */
    return rc;

  output_text("__CAPSTONE_SQLITE_EXTENDED_PASSED__\n");
  return SQLITE_OK;
}

static int run_sqlite(void) {
  sqlite3 *db = 0;
  sqlite3_stmt *statement = 0;
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

  rc = sqlite3_exec(db,
      "CREATE TABLE items(name TEXT NOT NULL, value INTEGER NOT NULL);",
      0, 0, 0);
  if (rc != SQLITE_OK)
    return fail("create", rc, db);

  rc = sqlite3_exec(db,
      "INSERT INTO items VALUES"
      "('alpha',11),('beta',22),('gamma',33);",
      0, 0, 0);
  if (rc != SQLITE_OK)
    return fail("insert", rc, db);

  rc = sqlite3_prepare_v2(
      db, "SELECT name,value FROM items;", -1, &statement, 0);
  if (rc != SQLITE_OK)
    return fail("prepare", rc, db);

  static const int expected_values[] = {11, 22, 33};
  unsigned row = 0;
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    const unsigned char *name = sqlite3_column_text(statement, 0);
    int value = sqlite3_column_int(statement, 1);
    const char *expected_name = 0;
    if (row == 0)
      expected_name = "alpha";
    else if (row == 1)
      expected_name = "beta";
    else if (row == 2)
      expected_name = "gamma";
    if (row >= 3 || !name ||
        sqlite3_stricmp((const char *)name, expected_name) != 0 ||
        value != expected_values[row])
      return fail("row-value", SQLITE_MISMATCH, db);
    output_text("row name=");
    output_text((const char *)name);
    output_text(" value=");
    output_uint((unsigned)value);
    output_text("\n");
    ++row;
  }

  if (rc != SQLITE_DONE || row != 3)
    return fail("step", rc, db);
  rc = sqlite3_finalize(statement);
  statement = 0;
  if (rc != SQLITE_OK)
    return fail("finalize", rc, db);

  /* Extended workload (indexes, joins, aggregates, mutations, string funcs) to
   * exercise more capability-sensitive SQLite machinery. */
  rc = run_sqlite_extended(db);
  if (rc != SQLITE_OK)
    return rc;

  rc = sqlite3_close(db);
  db = 0;
  if (rc != SQLITE_OK)
    return fail("close", rc, 0);

  output_text("__CAPSTONE_SQLITE_MEMORY_PASSED__\n");
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
  (void)run_sqlite();
  *res = SQLITE_HC_RET_DONE;
}
