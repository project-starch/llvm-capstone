#include <stdio.h>
#include "sqlite3.h"

int main(void) {
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;
  int rc;

  sqlite3_open(":memory:", &db);
  sqlite3_prepare_v2(db, "SELECT ?1", -1, &stmt, 0);
  rc = sqlite3_bind_text(stmt, 2, "wrong index", -1, SQLITE_STATIC);
  printf("bind rc=%d (%s)\n", rc, sqlite3_errstr(rc));
  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return rc == SQLITE_RANGE ? 0 : 1;
}
