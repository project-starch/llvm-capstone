#include <stdlib.h>
#include "sqlite3.h"

struct host_stmt {
  sqlite3 *db;
  sqlite3_stmt *stmt;
};

int main(void) {
  struct host_stmt *host = malloc(sizeof(*host));

  sqlite3_open(":memory:", &host->db);
  sqlite3_prepare_v2(host->db, "SELECT 1", -1, &host->stmt, 0);
  sqlite3_close_v2(host->db);
  free(host);

  sqlite3_step(host->stmt);
  return 0;
}
