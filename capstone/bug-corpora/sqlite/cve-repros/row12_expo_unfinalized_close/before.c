#include <stdlib.h>
#include "sqlite3.h"

struct connection {
  sqlite3 *db;
  sqlite3_stmt *stmt;
};

int main(void) {
  struct connection *connection = calloc(1, sizeof(*connection));
  volatile unsigned char byte;

  sqlite3_open(":memory:", &connection->db);
  sqlite3_prepare_v2(connection->db, "SELECT 1", -1,
                     &connection->stmt, 0);
  if (sqlite3_close(connection->db) != SQLITE_BUSY)
    return 1;

  connection->db = 0;
  byte = *(unsigned char *)connection->db;
  return byte;
}
