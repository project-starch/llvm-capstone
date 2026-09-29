#include <stdlib.h>
#include "sqlite3.h"

struct connection {
  sqlite3 *db;
};

int main(void) {
  sqlite3 *dst = 0;
  sqlite3_backup *backup;
  struct connection *src = malloc(sizeof(*src));

  sqlite3_open(":memory:", &src->db);
  sqlite3_open(":memory:", &dst);
  sqlite3_close_v2(src->db);
  free(src);

  backup = sqlite3_backup_init(dst, "main", src->db, "main");
  if (backup)
    sqlite3_backup_finish(backup);
  sqlite3_close(dst);
  return 0;
}
