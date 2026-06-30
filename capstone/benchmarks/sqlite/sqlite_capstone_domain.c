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
