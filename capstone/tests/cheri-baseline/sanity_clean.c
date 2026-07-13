/* No-defect SQLite exercise: open, create, insert, query, read a column, close.
 * Under CHERI purecap this MUST run to completion (exit 0) if the SQLite build
 * is capability-clean. If it faults, the amalgamation itself is not purecap-safe
 * and the corpus results would be measuring that, not the injected defect. */
#include <stdio.h>
#include "sqlite3.h"

int main(void) {
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;
  int val = -1;

  if (sqlite3_open(":memory:", &db) != SQLITE_OK) return 2;
  if (sqlite3_exec(db, "CREATE TABLE t(a); INSERT INTO t VALUES(123)",
                   0, 0, 0) != SQLITE_OK) return 3;
  if (sqlite3_prepare_v2(db, "SELECT a FROM t", -1, &stmt, 0) != SQLITE_OK)
    return 4;
  if (sqlite3_step(stmt) == SQLITE_ROW)
    val = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  printf("SANITY_OK val=%d\n", val);
  return (val == 123) ? 0 : 5;
}
