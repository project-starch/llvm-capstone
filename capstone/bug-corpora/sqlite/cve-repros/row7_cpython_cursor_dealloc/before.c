#include <stdlib.h>
#include "sqlite3.h"

struct cursor {
  sqlite3 *connection;
  sqlite3_stmt *statement;
};

int main(void) {
  struct cursor *cursor = malloc(sizeof(*cursor));

  sqlite3_open(":memory:", &cursor->connection);
  sqlite3_prepare_v2(cursor->connection, "SELECT 1", -1,
                     &cursor->statement, 0);
  sqlite3_close_v2(cursor->connection);
  free(cursor);

  sqlite3_step(cursor->statement);
  return 0;
}
