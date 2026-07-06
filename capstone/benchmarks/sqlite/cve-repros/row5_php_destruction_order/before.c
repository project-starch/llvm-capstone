#include <stdlib.h>
#include "sqlite3.h"

struct owner {
  sqlite3 *db;
  sqlite3_stmt *stmt;
};

int main(void) {
  struct owner *owner = malloc(sizeof(*owner));

  sqlite3_open(":memory:", &owner->db);
  sqlite3_prepare_v2(owner->db, "SELECT 1", -1, &owner->stmt, 0);
  sqlite3_close_v2(owner->db);
  free(owner);

  sqlite3_finalize(owner->stmt);
  return 0;
}
