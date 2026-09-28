#include <stdio.h>
#include "sqlite3.h"

int main(void) {
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;
  const char *name;

  sqlite3_open(":memory:", &db);
  sqlite3_exec(db, "CREATE TABLE t(a); INSERT INTO t VALUES(1)", 0, 0, 0);
  sqlite3_prepare_v2(db, "SELECT a AS colname FROM t", -1, &stmt, 0);
  sqlite3_step(stmt);
  name = sqlite3_column_name(stmt, 0);
  sqlite3_finalize(stmt);
  sqlite3_close(db);

  printf("%c\n", name[0]);
  return 0;
}
