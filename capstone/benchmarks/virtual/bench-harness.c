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
#ifdef BENCH_USER_COUNTERS
#include <setjmp.h>
#include <signal.h>
#include <sys/resource.h>
#endif

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

#ifdef BENCH_USER_COUNTERS
/* Plain arm (a native RISC-V Linux process, no capability context, no perf module). The counters
 * are the user-mode cycle and instret CSRs, read with rdcycle/rdinstret. They are accessible when
 * mcounteren and scounteren both have CY and IR set (OpenSBI sbi_hart.c writes SCOUNTEREN=7 and
 * MCOUNTEREN=-1; the board's Linux 6.4.14 has CONFIG_PERF_EVENTS off and never rewrites them). A
 * SIGILL-guarded probe at start-up decides: counters=present only if both reads work and
 * /proc/interrupts is readable, else counters=absent and every count reads 0. On the Capstone platforms
 * an illegal instruction goes to M first (medeleg does not delegate it). The board monitor of the
 * bitstream-3 perf kit (csbi-batch d9da397, handle_exception) prints ILLX/MTVL/MEPC on the console and,
 * for a trap from S or U, injects it into S (INJS; since f8bfde5, 2026-10-17), so the process gets
 * SIGILL and this guard prints counters=absent. The QEMU twin's monitor instead faults (cause 24 in
 * _return_from_domain, seen 2026-10-30 with -DBENCH_PROBE_TEST_TRAP) and halts the machine.
 *
 * Unlike the Capstone arms' CSRUNV windows, these counts INCLUDE kernel work inside the region:
 * timer ticks (HZ=100), any other interrupt, page faults and syscalls. To make that visible, each
 * bracket also snapshots /proc/interrupts (pread into a static buffer) and getrusage OUTSIDE the
 * counter reads, and `entries` is the number of asynchronous or fault entries into the kernel in the
 * window: delta(all interrupts) + delta(minor faults) + delta(major faults). The parts are printed
 * on a separate line after the BENCH line:
 *   BENCH-KERN <id> it=<k> timer=<d> irq=<d> minflt=<d> majflt=<d> nvcsw=<d> nivcsw=<d>
 * Order: begin = kernel snapshot, clock_gettime, rdcycle, rdinstret; end = rdinstret, rdcycle,
 * clock_gettime, kernel snapshot. So the counter bracket holds no syscall of the harness's own. */
static unsigned long long ucy[2], uin[2];
static char irqbuf[2][16384];
static long irqlen[2];
static struct rusage ru[2];
static int irq_fd = -1, user_ok;
static sigjmp_buf probe_jmp;
static void probe_sigill(int sig) { (void)sig; siglongjmp(probe_jmp, 1); }
static inline unsigned long long rd_cycle(void) {
  unsigned long long v;
  __asm__ volatile("rdcycle %0" : "=r"(v)::"memory");
  return v;
}
static inline unsigned long long rd_instret(void) {
  unsigned long long v;
  __asm__ volatile("rdinstret %0" : "=r"(v)::"memory");
  return v;
}
static int probe_user_counters(void) {
  struct sigaction sa, old;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = probe_sigill;
  sigemptyset(&sa.sa_mask);
  if (sigaction(SIGILL, &sa, &old))
    return 0;
  int ok = 0;
  if (!sigsetjmp(probe_jmp, 1)) {
#ifdef BENCH_PROBE_TEST_TRAP
    /* Negative control of the probe only: an illegal instruction where the counter reads are. */
    __asm__ volatile("unimp");
#endif
    volatile unsigned long long a = rd_cycle(), b = rd_instret(), c = rd_cycle(), d = rd_instret();
    ok = c > a && d > b;
  }
  sigaction(SIGILL, &old, 0);
  return ok;
}
static void kern_snapshot(int slot) {
  irqlen[slot] = irq_fd >= 0 ? pread(irq_fd, irqbuf[slot], sizeof irqbuf[slot] - 1, 0) : -1;
  getrusage(RUSAGE_SELF, &ru[slot]);
}
/* After the end mark only. /proc/interrupts: a header line of CPU names, then one line per source,
 * "<name>: <count per CPU> ... <description>". Sums every count of every line (all interrupts on
 * every CPU, IPIs included), and separately the line whose description contains riscv-timer. */
static void irq_parse(int slot, unsigned long long *total, unsigned long long *timer) {
  *total = *timer = 0;
  if (irqlen[slot] <= 0)
    return;
  char *s = irqbuf[slot], *end = s + irqlen[slot];
  *end = 0;
  char *line = strchr(s, '\n'); /* skip the CPU header */
  while (line && line < end) {
    line++;
    char *nl = strchr(line, '\n');
    if (!nl)
      nl = end;
    char *c = memchr(line, ':', (size_t)(nl - line));
    if (c) {
      unsigned long long sum = 0;
      char *q = c + 1;
      for (;;) {
        while (q < nl && *q == ' ')
          q++;
        if (q >= nl || *q < '0' || *q > '9')
          break;
        unsigned long long v = 0;
        while (q < nl && *q >= '0' && *q <= '9')
          v = v * 10 + (unsigned long long)(*q++ - '0');
        sum += v;
      }
      *total += sum;
      char save = *nl;
      *nl = 0;
      if (strstr(q, "riscv-timer"))
        *timer += sum;
      *nl = save;
    }
    line = nl < end ? nl : 0;
  }
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
#ifdef BENCH_USER_COUNTERS
  kern_snapshot(0);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  if (user_ok) {
    ucy[0] = rd_cycle();
    uin[0] = rd_instret();
  }
#else
  bench_mark(0);
  clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
}

void bench_end(void) {
#ifdef BENCH_USER_COUNTERS
  if (user_ok) {
    uin[1] = rd_instret();
    ucy[1] = rd_cycle();
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  kern_snapshot(1);
#else
  clock_gettime(CLOCK_MONOTONIC, &t1);
  bench_mark(1);
#endif
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
#ifdef BENCH_USER_COUNTERS
  irq_fd = open("/proc/interrupts", O_RDONLY);
  user_ok = probe_user_counters();
  present = user_ok && irq_fd >= 0 ? C_N : 0;
  printf("BENCH-INFO %s source=user-csr rdcycle_rdinstret=%s proc_interrupts=%s\n", id, user_ok ? "ok" : "trapped",
         irq_fd >= 0 ? "ok" : "absent");
  if (!user_ok)
    present = 0;
#else
  for (int k = 0; k < C_N; k++) {
    counter_fd[k] = open(counter_path[k], O_RDONLY);
    present += counter_fd[k] >= 0;
  }
#endif
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
#ifdef BENCH_USER_COUNTERS
    unsigned long long irq0, irq1, tim0, tim1;
    irq_parse(0, &irq0, &tim0);
    irq_parse(1, &irq1, &tim1);
    long minflt = ru[1].ru_minflt - ru[0].ru_minflt, majflt = ru[1].ru_majflt - ru[0].ru_majflt;
    long nvcsw = ru[1].ru_nvcsw - ru[0].ru_nvcsw, nivcsw = ru[1].ru_nivcsw - ru[0].ru_nivcsw;
    d[C_CYCLES] = user_ok ? ucy[1] - ucy[0] : 0;
    d[C_INSTRET] = user_ok ? uin[1] - uin[0] : 0;
    d[C_ENTRIES] = user_ok ? (irq1 - irq0) + (unsigned long long)(minflt + majflt) : 0;
#else
    for (int c = 0; c < C_N; c++)
      d[c] = parsed(1, c) - parsed(0, c);
#endif
    unsigned long long check = 0;
    int ok = post(&check);
    if (begins != b + 1 || ends != e + 1) {
      printf("BENCH-ERROR %s it=%s bracket begins=%d ends=%d (want exactly one each)\n", id, label,
             begins - b, ends - e);
      ok = 0;
    }
    printf("BENCH %s it=%s cycles=%llu instret=%llu entries=%llu ns=%lld check=0x%llx%s\n", id, label,
           d[C_CYCLES], d[C_INSTRET], d[C_ENTRIES], bench_window_ns(), check, ok ? "" : " VERIFY-FAILED");
#ifdef BENCH_USER_COUNTERS
    printf("BENCH-KERN %s it=%s timer=%llu irq=%llu minflt=%ld majflt=%ld nvcsw=%ld nivcsw=%ld\n", id, label,
           tim1 - tim0, irq1 - irq0, minflt, majflt, nvcsw, nivcsw);
#endif
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
