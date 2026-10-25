/* Reuse distance at SQLite's memsys5 boundary, observed from outside: linked into an unmodified
 * speedtest1 with
 *
 *   -Wl,--wrap=sqlite3_config -Wl,--wrap=sqlite3_step -Wl,--wrap=sqlite3_reset
 *
 * The program's own SQLITE_CONFIG_HEAP call reaches __wrap_sqlite3_config, which forwards every
 * configuration call unchanged (typed, per operation: sqlite3_config is variadic and has no va_list
 * form; an operation it does not know aborts the program, never forwarded wrongly). With lld, --wrap
 * also redirects SQLite's own calls inside sqlite3.c (e.g. PCACHE2 from sqlite3PCacheSetDefault, and
 * sqlite3_step from sqlite3_exec), which GNU ld leaves alone; SQL-CHECK then counts those too. After a successful HEAP call with a heap, it reads the installed methods (memsys5's) with
 * GETMALLOC and installs a copy whose xMalloc, xFree and xRealloc call memsys5's and record events;
 * xSize, xRoundup, xInit, xShutdown and pAppData are memsys5's own. Nothing here allocates, rounds,
 * retries or reads an allocation's contents, and no pointer is kept: only addresses, as integers.
 *
 * Events, in memsys5's own order:
 *   xMalloc(n)           attempt; if it returns p: issue(p)
 *   xFree(p)             release(p), then memsys5's free (the application's release: a quarantine
 *                        inside memsys5, if any, is part of the gap)
 *   xRealloc(p, n) -> q  q == p: in place, no event; q == 0: attempt, failed; else: attempt,
 *                        issue(q), release(p) (memsys5 allocates the new block, copies, frees the old)
 * Distance of a reuse = attempts at the moment of the reissue minus attempts at the release of the
 * same start; an immediate reissue is 1. bins[b] counts distances of bit length b+1, as
 * experiments/study/reuse-gap-observer.h does.
 *
 * The table holds one 32-bit word per 16-byte unit of the configured heap (static, zero pages until
 * touched): 0 never issued, LIVE while issued, else the attempt count at its release. A start outside
 * the heap, a misaligned start, an issue of a live start or a release of a start not live sets error.
 *
 * At exit, on stderr:
 *   REUSE-GAP memsys5 at=exit installed=1 heap=.. unit=16 attempts=.. issues=.. releases=.. reuses=..
 *     distinct=.. failed=.. moved=.. inplace=.. live=.. maxunit=.. error=.. first_error=..
 *     bins=b0,..,b31
 *   SQL-CHECK step_row=.. step_done=.. step_other=.. first_step_other=.. reset_ok=.. reset_err=..
 *     first_reset_err=..
 * The run is admissible only with installed=1, error=0, attempts == issues + failed,
 * issues == distinct + reuses, sum(bins) == reuses and live == issues - releases.
 */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "sqlite3.h"

#ifndef REUSE_HEAP_MAX
#define REUSE_HEAP_MAX (256u << 20)  /* their main() configures 256 MiB */
#endif
#define REUSE_UNIT_SHIFT 4
#define REUSE_UNITS (REUSE_HEAP_MAX >> REUSE_UNIT_SHIFT)
#define REUSE_LIVE UINT32_C(0xffffffff)

#if defined(__CHERI__)
#define REUSE_ADDR(p) ((uint64_t)__builtin_cheri_address_get(p))
#else
#define REUSE_ADDR(p) ((uint64_t)(uintptr_t)(p))
#endif

int __real_sqlite3_config(int op, ...);
int __real_sqlite3_step(sqlite3_stmt *);
int __real_sqlite3_reset(sqlite3_stmt *);

static uint32_t reuse_table[REUSE_UNITS];
static sqlite3_mem_methods reuse_inner, reuse_outer;
static uint64_t heap_base, heap_bytes;
static uint64_t attempts, issues, releases, reuses, distinct, failed, moved, inplace, maxunit;
static uint64_t bins[32];
static unsigned error;
static char first_error[96];
static int installed;
static uint64_t step_row, step_done, step_other, reset_ok, reset_err;
static int first_step_other, first_reset_err;

static void fail(unsigned code, const char *what, uint64_t a) {
  if (!error) {
    error = code;
    snprintf(first_error, sizeof first_error, "%s:0x%llx@%llu", what, (unsigned long long)a,
             (unsigned long long)attempts);
  }
}

static uint32_t *slot(uint64_t a, const char *what) {
  uint64_t off = a - heap_base;
  if (a < heap_base || off >= heap_bytes || off >= REUSE_HEAP_MAX) { fail(4, what, a); return 0; }
  if (off & ((1u << REUSE_UNIT_SHIFT) - 1)) { fail(5, what, a); return 0; }
  off >>= REUSE_UNIT_SHIFT;
  if (off > maxunit) maxunit = off;
  return &reuse_table[off];
}

#ifdef REUSE_WRAP_TRACE  /* tests only: one line per event, offsets from the heap */
#define TRACE(k, a) fprintf(stderr, "EV %c %llx %llu\n", k, (unsigned long long)((a) - heap_base), \
                            (unsigned long long)attempts)
#else
#define TRACE(k, a) ((void)0)
#endif

static void issue(uint64_t a) {
  uint32_t *s = slot(a, "issue");
  TRACE('I', a);
  if (!s) return;
  if (*s == REUSE_LIVE) { fail(2, "issue-live", a); return; }
  if (*s == 0) {
    ++distinct;
  } else {
    uint64_t gap = attempts - *s;
    unsigned b = 0;
    if (!gap) { fail(2, "issue-gap0", a); return; }
    while (b < 31 && gap > (UINT64_C(1) << (b + 1)) - 1) ++b;
    ++bins[b];
    ++reuses;
  }
  *s = REUSE_LIVE;
  ++issues;
}

static void release(uint64_t a) {
  uint32_t *s = slot(a, "release");
  TRACE('F', a);
  if (!s) return;
  if (*s != REUSE_LIVE) { fail(3, "release-not-live", a); return; }
  if (attempts >= REUSE_LIVE) { fail(6, "clock-overflow", a); return; }
  *s = (uint32_t)attempts;
  ++releases;
}

static void *outer_malloc(int n) {
  void *p = reuse_inner.xMalloc(n);
  ++attempts;
  if (p) issue(REUSE_ADDR(p)); else ++failed;
  return p;
}

static void outer_free(void *p) {
  if (p) release(REUSE_ADDR(p));
  reuse_inner.xFree(p);
}

static void *outer_realloc(void *p, int n) {
  uint64_t old = REUSE_ADDR(p);  /* before the call: the old block may be gone after it */
  void *q = reuse_inner.xRealloc(p, n);
  if (q && REUSE_ADDR(q) == old) { ++inplace; return q; }
  ++attempts;
  if (!q) { ++failed; return q; }
  ++moved;
  issue(REUSE_ADDR(q));
  release(old);
  return q;
}

/* The counters so far, labelled: at exit as "exit"; test-reuse-wrap.c also calls it mid-run. */
void reuse_wrap_snapshot(const char *label) {
  unsigned i;
  fprintf(stderr, "REUSE-GAP memsys5 at=%s installed=%d heap=%llu unit=%u attempts=%llu issues=%llu releases=%llu "
          "reuses=%llu distinct=%llu failed=%llu moved=%llu inplace=%llu live=%llu maxunit=%llu error=%u "
          "first_error=%s bins=", label, installed, (unsigned long long)heap_bytes, 1u << REUSE_UNIT_SHIFT,
          (unsigned long long)attempts, (unsigned long long)issues, (unsigned long long)releases,
          (unsigned long long)reuses, (unsigned long long)distinct, (unsigned long long)failed,
          (unsigned long long)moved, (unsigned long long)inplace,
          (unsigned long long)(issues - releases), (unsigned long long)maxunit, error,
          first_error[0] ? first_error : "-");
  for (i = 0; i < 32; ++i) fprintf(stderr, "%s%llu", i ? "," : "", (unsigned long long)bins[i]);
  fprintf(stderr, "\nSQL-CHECK step_row=%llu step_done=%llu step_other=%llu first_step_other=%d "
          "reset_ok=%llu reset_err=%llu first_reset_err=%d\n", (unsigned long long)step_row,
          (unsigned long long)step_done, (unsigned long long)step_other, first_step_other,
          (unsigned long long)reset_ok, (unsigned long long)reset_err, first_reset_err);
  fflush(stderr);
}

static void report(void) { reuse_wrap_snapshot("exit"); }

static int install(void *heap, int n) {
  int rc;
  if (installed) { fail(7, "second-heap", (uint64_t)n); return SQLITE_OK; }
  rc = __real_sqlite3_config(SQLITE_CONFIG_GETMALLOC, &reuse_inner);
  if (rc != SQLITE_OK) { fail(8, "getmalloc", (uint64_t)rc); return SQLITE_OK; }
  reuse_outer = reuse_inner;
  reuse_outer.xMalloc = outer_malloc;
  reuse_outer.xFree = outer_free;
  reuse_outer.xRealloc = outer_realloc;
  rc = __real_sqlite3_config(SQLITE_CONFIG_MALLOC, &reuse_outer);
  if (rc != SQLITE_OK) { fail(8, "malloc", (uint64_t)rc); return SQLITE_OK; }
  heap_base = REUSE_ADDR(heap);
  heap_bytes = (uint64_t)n;
  if (heap_bytes > REUSE_HEAP_MAX) fail(9, "heap-too-large", heap_bytes);
  installed = 1;
  if (atexit(report)) fail(10, "atexit", 0);
  return SQLITE_OK;
}

int __wrap_sqlite3_config(int op, ...) {
  va_list ap;
  int rc;
  va_start(ap, op);
  switch (op) {
  case SQLITE_CONFIG_SINGLETHREAD:
  case SQLITE_CONFIG_MULTITHREAD:
  case SQLITE_CONFIG_SERIALIZED:
    rc = __real_sqlite3_config(op);
    break;
  case SQLITE_CONFIG_MEMSTATUS:
  case SQLITE_CONFIG_URI:
  case SQLITE_CONFIG_COVERING_INDEX_SCAN:
  case SQLITE_CONFIG_STMTJRNL_SPILL:
  case SQLITE_CONFIG_SMALL_MALLOC: {
    int v = va_arg(ap, int);
    rc = __real_sqlite3_config(op, v);
    break;
  }
  case SQLITE_CONFIG_PMASZ: {
    unsigned v = va_arg(ap, unsigned);
    rc = __real_sqlite3_config(op, v);
    break;
  }
  /* One pointer, read and passed on with its own type (an ellipsis argument is not converted, so
   * va_arg must name the type the caller passed). SQLite calls some of these itself
   * (sqlite3PCacheSetDefault: PCACHE2): with lld, --wrap also redirects references inside the
   * object that defines the symbol. */
  case SQLITE_CONFIG_MALLOC:
  case SQLITE_CONFIG_GETMALLOC: {
    sqlite3_mem_methods *v = va_arg(ap, sqlite3_mem_methods *);
    if (op == SQLITE_CONFIG_MALLOC && installed) fail(11, "malloc-after-install", 0);
    rc = __real_sqlite3_config(op, v);
    break;
  }
  case SQLITE_CONFIG_MUTEX:
  case SQLITE_CONFIG_GETMUTEX: {
    sqlite3_mutex_methods *v = va_arg(ap, sqlite3_mutex_methods *);
    rc = __real_sqlite3_config(op, v);
    break;
  }
  case SQLITE_CONFIG_PCACHE:
  case SQLITE_CONFIG_GETPCACHE: {
    sqlite3_pcache_methods *v = va_arg(ap, sqlite3_pcache_methods *);
    rc = __real_sqlite3_config(op, v);
    break;
  }
  case SQLITE_CONFIG_PCACHE2:
  case SQLITE_CONFIG_GETPCACHE2: {
    sqlite3_pcache_methods2 *v = va_arg(ap, sqlite3_pcache_methods2 *);
    rc = __real_sqlite3_config(op, v);
    break;
  }
  case SQLITE_CONFIG_PCACHE_HDRSZ: {
    int *v = va_arg(ap, int *);
    rc = __real_sqlite3_config(op, v);
    break;
  }
  case SQLITE_CONFIG_LOG: {
    void (*f)(void *, int, const char *) = va_arg(ap, void (*)(void *, int, const char *));
    void *arg = va_arg(ap, void *);
    rc = __real_sqlite3_config(op, f, arg);
    break;
  }
  case SQLITE_CONFIG_SQLLOG: {
    void (*f)(void *, sqlite3 *, const char *, int) = va_arg(ap, void (*)(void *, sqlite3 *, const char *, int));
    void *arg = va_arg(ap, void *);
    rc = __real_sqlite3_config(op, f, arg);
    break;
  }
  case SQLITE_CONFIG_MMAP_SIZE: {
    sqlite3_int64 d = va_arg(ap, sqlite3_int64), m = va_arg(ap, sqlite3_int64);
    rc = __real_sqlite3_config(op, d, m);
    break;
  }
  case SQLITE_CONFIG_LOOKASIDE: {
    int sz = va_arg(ap, int), cnt = va_arg(ap, int);
    rc = __real_sqlite3_config(op, sz, cnt);
    break;
  }
  case SQLITE_CONFIG_HEAP:
  case SQLITE_CONFIG_PAGECACHE:
  case SQLITE_CONFIG_SCRATCH: {
    void *buf = va_arg(ap, void *);
    int n = va_arg(ap, int), m = va_arg(ap, int);
    rc = __real_sqlite3_config(op, buf, n, m);
    if (op == SQLITE_CONFIG_HEAP && rc == SQLITE_OK && buf) install(buf, n);
    /* A HEAP with no buffer after installation clears SQLite's methods: later allocations would
     * bypass the observer, so the measurement is void. */
    if (op == SQLITE_CONFIG_HEAP && rc == SQLITE_OK && !buf && installed) fail(12, "heap-null-after-install", 0);
    break;
  }
  default:
    va_end(ap);
    fprintf(stderr, "REUSE-GAP unsupported sqlite3_config op %d: the wrapper must forward it typed\n", op);
    abort();
  }
  va_end(ap);
  return rc;
}

int __wrap_sqlite3_step(sqlite3_stmt *s) {
  int rc = __real_sqlite3_step(s);
  if (rc == SQLITE_ROW) ++step_row;
  else if (rc == SQLITE_DONE) ++step_done;
  else { if (!step_other++) first_step_other = rc; }
  return rc;
}

int __wrap_sqlite3_reset(sqlite3_stmt *s) {
  int rc = __real_sqlite3_reset(s);
  if (rc == SQLITE_OK) ++reset_ok;
  else { if (!reset_err++) first_reset_err = rc; }
  return rc;
}
