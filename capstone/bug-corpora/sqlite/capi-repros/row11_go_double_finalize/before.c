#include "sqlite3.h"

int main(void) {
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;

  sqlite3_open(":memory:", &db);
  sqlite3_prepare_v2(db, "SELECT 1", -1, &stmt, 0);
  sqlite3_finalize(stmt);
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return 0;
}
