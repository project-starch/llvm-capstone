#include <stdio.h>
#include <stdlib.h>
#include "sqlite3.h"

static void read_context(sqlite3_context *sqlctx, int argc,
                         sqlite3_value **argv) {
  int value;
  (void)argc;
  (void)argv;
  value = *(int *)sqlite3_user_data(sqlctx);
  sqlite3_result_int(sqlctx, value);
}

int main(void) {
  sqlite3 *db = 0;
  int *app = malloc(sizeof(*app));
  char *error = 0;

  *app = 42;
  sqlite3_open(":memory:", &db);
  sqlite3_create_function(db, "read_context", 0, SQLITE_UTF8, app,
                          read_context, 0, 0);
  free(app);
  sqlite3_exec(db, "SELECT read_context()", 0, 0, &error);
  fprintf(stderr, "%s\n", error ? error : "query returned");
  sqlite3_free(error);
  sqlite3_close(db);
  return 0;
}
