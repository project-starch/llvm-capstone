#include <stdlib.h>
#include "sqlite3.h"

struct udf_state {
  int result;
};

static void stale_udf(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  struct udf_state *state = sqlite3_user_data(ctx);
  (void)argc;
  (void)argv;
  sqlite3_result_int(ctx, state->result);
}

int main(void) {
  sqlite3 *db = 0;
  struct udf_state *state = malloc(sizeof(*state));

  state->result = 99;
  sqlite3_open(":memory:", &db);
  sqlite3_create_function(db, "stale_udf", 0, SQLITE_UTF8, state,
                          stale_udf, 0, 0);
  free(state);
  sqlite3_exec(db, "SELECT stale_udf()", 0, 0, 0);
  sqlite3_close(db);
  return 0;
}
