#include <stdio.h>
#include <stdlib.h>
#include "sqlite3.h"

struct progress_ctx {
  sqlite3 *db;
  int calls;
};

static int progress(void *opaque) {
  struct progress_ctx *ctx = opaque;
  sqlite3_progress_handler(ctx->db, 0, 0, 0);
  free(ctx);
  return 0;
}

int main(void) {
  sqlite3 *db = 0;
  struct progress_ctx *ctx;
  volatile int observed;

  if (sqlite3_open(":memory:", &db) != SQLITE_OK)
    return 1;
  ctx = malloc(sizeof(*ctx));
  ctx->db = db;
  ctx->calls = 7;
  sqlite3_progress_handler(db, 1, progress, ctx);
  sqlite3_exec(db, "SELECT 1 UNION ALL SELECT 2", 0, 0, 0);

  observed = ctx->calls;
  printf("%d\n", observed);
  sqlite3_close(db);
  return 0;
}
