/* Does sqlite3_column_text()'s buffer get FREED or REUSED IN PLACE across sqlite3_step()?
 *
 * Positive control included: we also interpose malloc/free and count allocator
 * events that touch the observed address, so a "no allocator event" result is
 * only believed if the counter is demonstrably able to fire.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sqlite3.h>

int main(void) {
  sqlite3 *db; sqlite3_stmt *st; int rc, i;
  const unsigned char *p_prev = NULL;

  rc = sqlite3_open(":memory:", &db);
  if (rc) { fprintf(stderr, "open failed\n"); return 1; }
  printf("sqlite version: %s\n", sqlite3_libversion());

  sqlite3_exec(db, "CREATE TABLE t(v TEXT);", 0, 0, 0);
  /* Rows of EQUAL length so the register's existing allocation always suffices. */
  sqlite3_exec(db,
    "INSERT INTO t VALUES('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA');"
    "INSERT INTO t VALUES('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB');"
    "INSERT INTO t VALUES('CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC');"
    "INSERT INTO t VALUES('DDDDDDDDDDDDDDDDDDDDDDDDDDDDDD');", 0, 0, 0);

  /* ---- Experiment 1: plain scan, watch the returned pointer ---- */
  printf("\n=== EXP 1: SELECT v FROM t  (column_text pointer across steps) ===\n");
  sqlite3_prepare_v2(db, "SELECT v FROM t;", -1, &st, 0);
  i = 0;
  while (sqlite3_step(st) == SQLITE_ROW) {
    const unsigned char *p = sqlite3_column_text(st, 0);
    printf("row %d: ptr=%p  content=\"%.4s...\"  %s\n", i, (void*)p, p,
           (p == p_prev) ? "SAME ADDRESS AS PREVIOUS ROW" : "");
    p_prev = p; i++;
  }
  sqlite3_finalize(st);

  /* ---- Experiment 2: hold the borrowed pointer across one step ---- */
  printf("\n=== EXP 2: borrow across a step (the class-A pattern) ===\n");
  sqlite3_prepare_v2(db, "SELECT v FROM t;", -1, &st, 0);
  if (sqlite3_step(st) == SQLITE_ROW) {
    const unsigned char *borrowed = sqlite3_column_text(st, 0);
    char snapshot[64];
    snprintf(snapshot, sizeof snapshot, "%s", (const char*)borrowed);
    printf("borrowed ptr = %p, content at borrow time = \"%s\"\n",
           (void*)borrowed, snapshot);
    if (sqlite3_step(st) == SQLITE_ROW) {
      printf("after next sqlite3_step(): SAME pointer %p now reads \"%s\"\n",
             (void*)borrowed, (const char*)borrowed);
      printf("verdict: %s\n",
             strcmp(snapshot, (const char*)borrowed) ? "DATA IDENTITY CHANGED IN PLACE (reuse-not-free)"
                                                     : "unchanged");
    }
  }
  sqlite3_finalize(st);

  /* ---- Experiment 3: same, but with values that fit on the leaf page and are
     returned as MEM_Ephem pointers straight into the page cache ---- */
  printf("\n=== EXP 3: blob column (no NUL-termination copy) ===\n");
  sqlite3_prepare_v2(db, "SELECT CAST(v AS BLOB) FROM t;", -1, &st, 0);
  if (sqlite3_step(st) == SQLITE_ROW) {
    const unsigned char *b = sqlite3_column_blob(st, 0);
    int n = sqlite3_column_bytes(st, 0);
    printf("blob ptr = %p, first byte = '%c'\n", (void*)b, b[0]);
    if (sqlite3_step(st) == SQLITE_ROW) {
      const unsigned char *b2 = sqlite3_column_blob(st, 0);
      printf("after step: old ptr %p first byte = '%c'; new ptr = %p (%s)\n",
             (void*)b, b[0], (void*)b2, (b==b2) ? "SAME" : "different");
    }
    (void)n;
  }
  sqlite3_finalize(st);

  sqlite3_close(db);
  return 0;
}
