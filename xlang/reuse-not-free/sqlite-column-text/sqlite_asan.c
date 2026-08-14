/* Positive control for the "no allocator event" claim.
 *
 * Arm A (control): a genuine free-then-use. ASan MUST report heap-use-after-free.
 * Arm B (subject) : sqlite3_column_text() borrowed across sqlite3_step().
 *
 * If ASan fires on A and is silent on B, then B is not a free-then-use.
 * Run: ./sqlite_asan A   and   ./sqlite_asan B
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sqlite3.h>

static void arm_control(void) {
  char *p = strdup("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA");
  printf("control: ptr=%p content=%.4s\n", (void*)p, p);
  free(p);
  printf("control: after free, reads '%c' (ASan should have aborted)\n", p[0]);
}

static void arm_sqlite(void) {
  sqlite3 *db; sqlite3_stmt *st;
  sqlite3_open(":memory:", &db);
  sqlite3_exec(db, "CREATE TABLE t(v TEXT);", 0, 0, 0);
  sqlite3_exec(db,
    "INSERT INTO t VALUES('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA');"
    "INSERT INTO t VALUES('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB');", 0, 0, 0);
  sqlite3_prepare_v2(db, "SELECT v FROM t;", -1, &st, 0);
  sqlite3_step(st);
  const unsigned char *borrowed = sqlite3_column_text(st, 0);
  printf("sqlite: borrowed ptr=%p content=%.4s\n", (void*)borrowed, borrowed);
  sqlite3_step(st);                       /* the loan ends here */
  printf("sqlite: after step, same ptr %p reads %.4s\n",
         (void*)borrowed, borrowed);      /* the bug: reads row 2's data */
  sqlite3_finalize(st);
  sqlite3_close(db);
}

int main(int argc, char **argv) {
  if (argc > 1 && argv[1][0] == 'A') arm_control(); else arm_sqlite();
  return 0;
}
