#include <stdlib.h>
#include "sqlite3.h"

struct ruby_statement {
  sqlite3 *database;
  sqlite3_stmt *statement;
};

int main(void) {
  struct ruby_statement *wrapper = malloc(sizeof(*wrapper));

  sqlite3_open(":memory:", &wrapper->database);
  sqlite3_prepare_v2(wrapper->database, "SELECT 1", -1,
                     &wrapper->statement, 0);
  sqlite3_close_v2(wrapper->database);
  free(wrapper);

  sqlite3_finalize(wrapper->statement);
  return 0;
}
