#include <stdio.h>
#include "sqlite3.h"

int main(void) {
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;
  const unsigned char *borrowed;

  sqlite3_open(":memory:", &db);
  sqlite3_prepare_v2(db, "SELECT 'alpha' UNION ALL SELECT 'beta'",
                     -1, &stmt, 0);
  sqlite3_step(stmt);
  borrowed = sqlite3_column_text(stmt, 0);
  sqlite3_step(stmt);
  printf("borrowed-after-restep=%s current=%s\n", borrowed,
         sqlite3_column_text(stmt, 0));
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return 0;
}
