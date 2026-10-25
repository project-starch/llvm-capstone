/* Test program for reuse-wrap.c: configures memsys5 as PoisonCap's speedtest1 main() does
 * (SINGLETHREAD, HEAP with minimum 16, MEMSTATUS 0), then
 *   1. a fixed pseudo-random sequence of sqlite3_malloc / sqlite3_free / sqlite3_realloc, printing
 *      every call and every returned address ("M n addr", "F addr", "R old n new"), so that
 *      check-reuse-wrap.py can recompute the wrapper's counters and bins independently, and so that
 *      the address sequence of a build with the wrapper can be compared with one without it;
 *   2. SQL through prepared statements, with one deliberate UNIQUE violation: sqlite3_step must return
 *      SQLITE_CONSTRAINT once and the following sqlite3_reset must return it once (the SQL-CHECK
 *      line's positive control).
 * The heap comes from malloc, as in their main() (a static array needs bounds alignment the Sublet port
 * refuses to derive from). Unwrapped builds link snapshot-stub.c: a weak undefined function is not null
 * on every target (on Capstone the call lands outside the code), so the program never tests for it. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "sqlite3.h"

#define HEAP_BYTES (8u << 20)
static unsigned char *heap;

static uint64_t rng = 88172645463325252ull;
static unsigned next(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (unsigned)rng; }

#if defined(__CHERI__)
#define ADDR(p) ((unsigned long long)__builtin_cheri_address_get(p))
#else
#define ADDR(p) ((unsigned long long)(uintptr_t)(p))
#endif
/* Offsets from the heap, so that runs of different binaries (and address randomisation) compare;
 * a failed call prints "null" (offset 0 is a real block). */
static char obuf[4][24];
static int oi;
static const char *off(const void *p) {
  char *b = obuf[oi++ & 3];
  if (!p) return "null";
  snprintf(b, sizeof obuf[0], "%llx", ADDR(p) - ADDR(heap));
  return b;
}

void reuse_wrap_snapshot(const char *label);  /* reuse-wrap.c, or snapshot-stub.c in unwrapped builds */

int main(int argc, char **argv) {
  void *live[256] = {0};
  int i, rc;
  /* Steps of the call sequence (default 20000). On PoisonCap the program's own live pointer is changed
   * from outside after about 6,000 steps (2026-10-25, with and without the wrapper, with and without
   * libc revocation): its qualification uses a shorter sequence. */
  int steps = argc > 1 ? atoi(argv[1]) : 20000;
  sqlite3 *db;
  sqlite3_stmt *st;
  heap = malloc(HEAP_BYTES);
  if (!heap) { puts("heap malloc failed"); return 1; }
  sqlite3_shutdown();
  sqlite3_config(SQLITE_CONFIG_SINGLETHREAD);
  if (sqlite3_config(SQLITE_CONFIG_HEAP, heap, HEAP_BYTES, 16) != SQLITE_OK) { puts("HEAP failed"); return 1; }
  sqlite3_config(SQLITE_CONFIG_MEMSTATUS, 0);
  sqlite3_initialize();
  puts("BASE");
  fflush(stdout);
  reuse_wrap_snapshot("start");
  for (i = 0; i < steps; ++i) {
    unsigned k = next() % 256, op = next() % 4, n = 1 + next() % 3000;
    if (!live[k] && op != 3) {
      live[k] = sqlite3_malloc((int)n);
      printf("M %u %s\n", n, off(live[k]));
    } else if (live[k] && op == 3) {
      char was[24];  /* the old offset, taken before the call: the old block may be gone after it */
      snprintf(was, sizeof was, "%s", off(live[k]));
      void *q = sqlite3_realloc(live[k], (int)n);
      printf("R %s %u %s\n", was, n, off(q));
      if (q) live[k] = q;
    } else if (live[k]) {
      printf("F %s\n", off(live[k]));
      sqlite3_free(live[k]);
      live[k] = 0;
    }
  }
  for (i = 0; i < 256; ++i) if (live[i]) { printf("F %s\n", off(live[i])); sqlite3_free(live[i]); }
  puts("ALLOC-DONE");
  fflush(stdout);
  reuse_wrap_snapshot("alloc-done");
  /* SQL: allocations from here on are not printed; the oracle covers the "alloc-done" snapshot. */
  if (sqlite3_open(":memory:", &db)) { puts("open failed"); return 1; }
  sqlite3_exec(db, "CREATE TABLE t(a UNIQUE)", 0, 0, 0);
  sqlite3_prepare_v2(db, "INSERT INTO t VALUES(?1)", -1, &st, 0);
  for (i = 0; i < 5; ++i) {
    sqlite3_bind_int(st, 1, i == 4 ? 0 : i);  /* the fifth insert repeats 0: SQLITE_CONSTRAINT */
    rc = sqlite3_step(st);
    printf("STEP %d\n", rc);
    printf("RESET %d\n", sqlite3_reset(st));
  }
  sqlite3_finalize(st);
  sqlite3_close(db);
  puts("SQL-DONE");
  return 0;
}
