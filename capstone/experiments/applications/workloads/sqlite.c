#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void phase(const char *name) {
  char line[160];
  int len = snprintf(line, sizeof line, "MEMPHASE %s\n", name);
  write(2, line, len);
  len = snprintf(line, sizeof line, "EXP-INNER phase=%s live=%lld peak=%lld\n",
                 name, sqlite3_memory_used(), sqlite3_memory_highwater(0));
  write(2, line, len);
}
static void sql(sqlite3 *db, const char *text) {
  int rc = sqlite3_exec(db, text, 0, 0, 0);
  if (rc) { fprintf(stderr, "sql error %d: %s\n", rc, sqlite3_errmsg(db)); exit(1); }
}
static sqlite3 *database(int rows) {
  sqlite3 *db = 0;
  if (sqlite3_open(":memory:", &db)) exit(2);
  if (sqlite3_db_config(db, SQLITE_DBCONFIG_LOOKASIDE, 0, 1200, 40)) exit(3);
  sql(db, "CREATE TABLE records(id INTEGER PRIMARY KEY, value TEXT); BEGIN;");
  sqlite3_stmt *s = 0;
  if (sqlite3_prepare_v2(db, "INSERT INTO records VALUES(?1, ?2)", -1, &s, 0)) exit(4);
  for (int i = 0; i < rows; ++i) {
    char text[128]; memset(text, 'a' + i % 26, 96); text[96] = 0;
    sqlite3_bind_int(s, 1, i);
    sqlite3_bind_text(s, 2, text, -1, SQLITE_TRANSIENT);
    if (sqlite3_step(s) != SQLITE_DONE) exit(5);
    sqlite3_reset(s);
  }
  sqlite3_finalize(s);
  sql(db, "COMMIT; CREATE INDEX by_value ON records(value);");
  return db;
}
static long long checksum(sqlite3 *db) {
  sqlite3_stmt *s = 0;
  if (sqlite3_prepare_v2(db, "SELECT id FROM records ORDER BY value, id DESC", -1, &s, 0)) exit(6);
  long long sum = 0; int rc;
  while ((rc = sqlite3_step(s)) == SQLITE_ROW) sum += sqlite3_column_int64(s, 0);
  if (rc != SQLITE_DONE) exit(7);
  sqlite3_finalize(s);
  return sum;
}
int main(int argc, char **argv) {
  if (argc != 4) return 64;
  int n = atoi(argv[1]), batches = atoi(argv[2]), retained = atoi(argv[3]);
  if (n < 1 || batches < 1 || retained < 0) return 64;
  void *heap = malloc(8 << 20);
  if (!heap || sqlite3_config(SQLITE_CONFIG_HEAP, heap, 8 << 20, 32) ||
      sqlite3_config(SQLITE_CONFIG_MEMSTATUS, 1) || sqlite3_initialize()) return 8;
  sqlite3 *keep = database(retained);
  phase("baseline");
  long long total = 0;
  for (int e = 0; e < batches; ++e) {
    sqlite3 *db = database(n * (e == batches/2 ? 4 : 1));
    total += checksum(db);
    char label[64]; snprintf(label, sizeof label, "live-%d", e); phase(label);
    if (sqlite3_close(db)) return 9;
    snprintf(label, sizeof label, "released-%d", e); phase(label);
  }
  if (checksum(keep) != (long long)retained*(retained-1)/2) return 10;
  if (sqlite3_close(keep) || sqlite3_shutdown()) return 11;
  free(heap);
  long long burst = 4LL*n;
  if (total != (batches-1LL)*n*(n-1)/2 + burst*(burst-1)/2) return 12;
  printf("EXP-OK sqlite %lld\n", total);
  return 0;
}
