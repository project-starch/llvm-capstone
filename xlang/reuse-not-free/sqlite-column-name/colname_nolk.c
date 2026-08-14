/* RUSTSEC-2021-0037 shape: is the sqlite3_column_name() pointer invalidated by
 * an automatic reprepare a FREE-then-use, or a reuse-in-place?
 * Conn B changes the schema, which forces conn A's next sqlite3_step() to
 * automatically reprepare the statement.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sqlite3.h>

int main(void) {
  sqlite3 *a, *b; sqlite3_stmt *st;
  const char *dbfile = "/tmp/colname.db";
  remove(dbfile);
  sqlite3_config(SQLITE_CONFIG_LOOKASIDE, 0, 0); sqlite3_open(dbfile, &a);
  sqlite3_open(dbfile, &b);
  sqlite3_exec(a, "CREATE TABLE t(alpha TEXT, beta TEXT);", 0, 0, 0);
  sqlite3_exec(a, "INSERT INTO t VALUES('x','y');", 0, 0, 0);

  sqlite3_prepare_v2(a, "SELECT alpha, beta FROM t;", -1, &st, 0);
  const char *n0 = sqlite3_column_name(st, 0);
  const char *n1 = sqlite3_column_name(st, 1);
  char snap0[64]; snprintf(snap0, sizeof snap0, "%s", n0);
  printf("before step: name0 ptr=%p \"%s\"   name1 ptr=%p \"%s\"\n",
         (void*)n0, n0, (void*)n1, n1);

  /* force a schema change on the other connection -> automatic reprepare */
  int rc = sqlite3_exec(b, "CREATE TABLE zz(q);", 0, 0, 0);
  printf("schema change on conn B: rc=%d\n", rc);

  rc = sqlite3_step(st);
  printf("step rc=%d\n", rc);
  const char *n0b = sqlite3_column_name(st, 0);
  printf("after step: name0 ptr=%p \"%s\"  (%s)\n", (void*)n0b, n0b,
         (n0b == n0) ? "SAME ADDRESS as pre-step pointer" : "different address");
  printf("stale pointer %p now reads \"%s\"  -> %s\n", (void*)n0, n0,
         strcmp(snap0, n0) ? "CONTENT CHANGED" : "content unchanged");

  sqlite3_finalize(st); sqlite3_close(a); sqlite3_close(b);
  remove(dbfile);
  return 0;
}
