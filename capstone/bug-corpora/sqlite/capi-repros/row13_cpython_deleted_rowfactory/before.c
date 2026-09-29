#include <stdlib.h>
#include "sqlite3.h"

struct cursor {
  sqlite3 *db;
  int (*row_factory)(int);
};

static int make_row(int value) {
  return value;
}

int main(void) {
  struct cursor *cursor = malloc(sizeof(*cursor));

  sqlite3_open(":memory:", &cursor->db);
  cursor->row_factory = make_row;
  cursor->row_factory = 0;
  return cursor->row_factory(7);
}
