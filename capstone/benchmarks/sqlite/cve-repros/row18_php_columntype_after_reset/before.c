#include <stdio.h>
#include "sqlite3.h"

int main(void) {
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;
  int before;
  int after;

  sqlite3_open(":memory:", &db);
  sqlite3_prepare_v2(db, "SELECT 123", -1, &stmt, 0);
  sqlite3_step(stmt);
  before = sqlite3_column_type(stmt, 0);
  sqlite3_reset(stmt);
  after = sqlite3_column_type(stmt, 0);
  printf("before=%d after-reset=%d\n", before, after);
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return 0;
}
