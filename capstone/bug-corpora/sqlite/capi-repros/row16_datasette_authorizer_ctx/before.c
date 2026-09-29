#include <stdlib.h>
#include "sqlite3.h"

struct auth_ctx {
  int decision;
};

static int authorize(void *opaque, int action, const char *arg1,
                     const char *arg2, const char *database,
                     const char *trigger) {
  struct auth_ctx *ctx = opaque;
  (void)action;
  (void)arg1;
  (void)arg2;
  (void)database;
  (void)trigger;
  return ctx->decision;
}

int main(void) {
  sqlite3 *db = 0;
  struct auth_ctx *ctx = malloc(sizeof(*ctx));

  ctx->decision = SQLITE_OK;
  sqlite3_open(":memory:", &db);
  sqlite3_set_authorizer(db, authorize, ctx);
  free(ctx);
  sqlite3_exec(db, "CREATE TABLE protected(value)", 0, 0, 0);
  sqlite3_close(db);
  return 0;
}
