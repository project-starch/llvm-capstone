/* The shared measurement harness; see bench-harness.h for the protocol.
 *
 * A mark (bench_mark) only preads each counter file into a static buffer: no allocation, no stdio,
 * no parsing. The buffers are parsed after the iteration's end mark, and the line is printed after
 * that. The three files are opened once, at start-up, outside every bracket. */
#include "bench-harness.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { C_CYCLES, C_INSTRET, C_ENTRIES, C_N };
static const char *const counter_path[C_N] = {
    "/sys/module/capstone_vm/parameters/bench_cycles",
    "/sys/module/capstone_vm/parameters/bench_instret",
    "/sys/module/capstone_vm/parameters/bench_entries",
};
static const char *const counter_name[C_N] = {"cycles", "instret", "entries"};
static int counter_fd[C_N] = {-1, -1, -1};
/* raw[0] at the begin mark, raw[1] at the end mark */
static char raw[2][C_N][32];
static long raw_len[2][C_N];
static struct timespec t0, t1;
static int begins, ends;

#ifdef BENCH_COUNT_ALLOCS
/* Counting build only. The wrappers count a call at depth 0, so an allocator entry point that
 * reaches another wrapped one through an external reference is counted once. */
enum { K_MALLOC, K_FREE, K_REALLOC, K_CALLOC, K_N };
static unsigned long counts[K_N], at_begin[K_N], window[K_N];
static int depth;
void *__real_malloc(size_t);
void __real_free(void *);
void *__real_realloc(void *, size_t);
void *__real_calloc(size_t, size_t);
void *__wrap_malloc(size_t n) {
  if (depth++ == 0) counts[K_MALLOC]++;
  void *r = __real_malloc(n);
  depth--;
  return r;
}
void __wrap_free(void *p) {
  if (depth++ == 0) counts[K_FREE]++;
  __real_free(p);
  depth--;
}
void *__wrap_realloc(void *p, size_t n) {
  if (depth++ == 0) counts[K_REALLOC]++;
  void *r = __real_realloc(p, n);
  depth--;
  return r;
}
void *__wrap_calloc(size_t m, size_t n) {
  if (depth++ == 0) counts[K_CALLOC]++;
  void *r = __real_calloc(m, n);
  depth--;
  return r;
}
static void print_counts(const char *id, const char *label, const unsigned long *k) {
  printf("ALLOC %s it=%s malloc=%lu free=%lu realloc=%lu calloc=%lu\n", id, label, k[K_MALLOC],
         k[K_FREE], k[K_REALLOC], k[K_CALLOC]);
}
#endif

/* The mark: one pread per present counter into a static buffer. */
static void bench_mark(int slot) {
  for (int k = 0; k < C_N; k++)
    raw_len[slot][k] = counter_fd[k] >= 0 ? pread(counter_fd[k], raw[slot][k], sizeof raw[slot][k] - 1, 0) : -1;
}

/* After the end mark only. An absent or unreadable counter reads 0. */
static unsigned long long parsed(int slot, int k) {
  if (raw_len[slot][k] <= 0)
    return 0;
  unsigned long long v = 0;
  for (long i = 0; i < raw_len[slot][k] && raw[slot][k][i] >= '0' && raw[slot][k][i] <= '9'; i++)
    v = v * 10 + (unsigned long long)(raw[slot][k][i] - '0');
  return v;
}

void bench_begin(void) {
  begins++;
#ifdef BENCH_COUNT_ALLOCS
  memcpy(at_begin, counts, sizeof counts);
#endif
  bench_mark(0);
  clock_gettime(CLOCK_MONOTONIC, &t0);
}

void bench_end(void) {
  clock_gettime(CLOCK_MONOTONIC, &t1);
  bench_mark(1);
#ifdef BENCH_COUNT_ALLOCS
  for (int k = 0; k < K_N; k++)
    window[k] = counts[k] - at_begin[k];
#endif
  ends++;
}

long long bench_window_ns(void) {
  return (long long)(t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
}

unsigned long long bench_fnv(unsigned long long h, const void *p, unsigned long n) {
  const unsigned char *b = p;
  for (unsigned long i = 0; i < n; i++) {
    h ^= b[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static int parse_count(const char *s, int fallback) {
  if (!s)
    return fallback;
  char *end;
  long v = strtol(s, &end, 10);
  if (*end || v < 0 || v > 1000000) {
    fprintf(stderr, "bench: bad iteration count '%s'\n", s);
    exit(2);
  }
  return (int)v;
}

static int run(const char *id, int argc, char **argv, bench_unit_fn unit, bench_post_fn post,
               int self) {
  int warm = parse_count(argc > 1 ? argv[1] : 0, 1);
  int iters = parse_count(argc > 2 ? argv[2] : 0, 5);
  int present = 0;
  for (int k = 0; k < C_N; k++) {
    counter_fd[k] = open(counter_path[k], O_RDONLY);
    present += counter_fd[k] >= 0;
  }
  printf("BENCH-INFO %s counters=", id);
  if (present == C_N)
    printf("present");
  else if (present == 0)
    printf("absent");
  else
    for (int k = 0, first = 1; k < C_N; k++)
      if (counter_fd[k] >= 0) {
        printf("%spartial:%s", first ? "" : ",", counter_name[k]);
        first = 0;
      }
  printf(" warmup=%d iterations=%d\n", warm, iters);
  fflush(stdout);
  int failed = 0;
  for (int k = 0; k < warm + iters; k++) {
    char label[24];
    if (k < warm)
      snprintf(label, sizeof label, "w%d", k);
    else
      snprintf(label, sizeof label, "%d", k - warm);
    int b = begins, e = ends;
    if (!self)
      bench_begin();
    unit();
    if (!self)
      bench_end();
    unsigned long long d[C_N];
    for (int c = 0; c < C_N; c++)
      d[c] = parsed(1, c) - parsed(0, c);
    unsigned long long check = 0;
    int ok = post(&check);
    if (begins != b + 1 || ends != e + 1) {
      printf("BENCH-ERROR %s it=%s bracket begins=%d ends=%d (want exactly one each)\n", id, label,
             begins - b, ends - e);
      ok = 0;
    }
    printf("BENCH %s it=%s cycles=%llu instret=%llu entries=%llu ns=%lld check=0x%llx%s\n", id, label,
           d[C_CYCLES], d[C_INSTRET], d[C_ENTRIES], bench_window_ns(), check, ok ? "" : " VERIFY-FAILED");
#ifdef BENCH_COUNT_ALLOCS
    print_counts(id, label, window);
#endif
    fflush(stdout);
    if (!ok)
      failed++;
  }
#ifdef BENCH_COUNT_ALLOCS
  print_counts(id, "total", counts);
#endif
  printf("BENCH-DONE %s failed=%d\n", id, failed);
  fflush(stdout);
  return failed ? 1 : 0;
}

int bench_run(const char *id, int argc, char **argv, bench_unit_fn unit, bench_post_fn post) {
  return run(id, argc, argv, unit, post, 0);
}

int bench_run_self(const char *id, int argc, char **argv, bench_unit_fn unit, bench_post_fn post) {
  return run(id, argc, argv, unit, post, 1);
}
