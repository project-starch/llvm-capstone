/* Board probe for the Plain arm's instrument: user-mode rdcycle/rdinstret, and what a kernel entry
 * costs inside a counted region.
 *
 * 1. Access. rdcycle and rdinstret are read under a SIGILL handler. If either traps (scounteren or
 *    mcounteren lacks CY/IR on this platform), it prints `PLAINPROBE access=trapped` and exits 2:
 *    the Plain arm then has no counters on this board, and the kit stops reading Plain rows.
 *    On the board (monitor csbi-batch d9da397) the trap goes to M, the monitor prints ILLX, MTVL (the
 *    rdcycle/rdinstret word) and MEPC, injects it into S (INJS), and this branch prints access=trapped.
 *    An older monitor without injection would spin instead: a wedge right after @@PLAINPROBE-BEGIN.
 *    The QEMU twin's monitor halts the machine on the same trap.
 * 2. Monotonicity and rate: cycles and instret advance; cycles per ns over a ~1 s spin against
 *    CLOCK_MONOTONIC (prediction on the 25 MHz board: 0.025 cycles/ns).
 * 3. Gaps. A tight loop reads rdcycle and rdinstret back to back for SPIN_SECONDS. The loop's own
 *    iteration is a few instructions and cycles; an iteration that took more than GAP_CYCLES is a
 *    kernel entry (timer tick, other interrupt) that landed between two reads. Each gap's cycles and
 *    instret are recorded (no stdio inside the spin) and summarised after it: count, min, median,
 *    max. With HZ=100 and one hart, ~100 gaps per second are predicted. The median gap is the
 *    per-tick cost a Plain reading carries, EXCLUDING the cache and TLB refill the program pays after
 *    the tick (this loop's working set is a few lines), so it is a lower bound on the per-tick cost.
 *    /proc/interrupts is read before and after the spin and the delta printed beside the gap count:
 *    the two must agree within a few (the counter-based gap detector is checked against the
 *    kernel's own count).
 * 4. Faults. FAULT_PAGES fresh anonymous pages are mapped and each touched once by a store, with
 *    rdcycle/rdinstret around the store: each reading is one minor page fault (plus the store), the
 *    per-fault cost a Plain reading carries where a program first-touches memory inside its region
 *    (the Capstone arms take the same fault as an escape, outside their window). The getrusage
 *    minor-fault delta is printed beside it as the check that every touch faulted.
 * No allocation and no stdio inside any timed part. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef SPIN_SECONDS
#define SPIN_SECONDS 2
#endif
#ifndef GAP_CYCLES
#define GAP_CYCLES 400
#endif
#define MAXGAP 4096
#ifndef FAULT_PAGES
#define FAULT_PAGES 256
#endif
static unsigned long long f_cy[FAULT_PAGES], f_in[FAULT_PAGES];

static sigjmp_buf jb;
static void on_sigill(int s) { (void)s; siglongjmp(jb, 1); }
static inline unsigned long long rdc(void) { unsigned long long v; __asm__ volatile("rdcycle %0" : "=r"(v)::"memory"); return v; }
static inline unsigned long long rdi(void) { unsigned long long v; __asm__ volatile("rdinstret %0" : "=r"(v)::"memory"); return v; }

static unsigned long long gap_cy[MAXGAP], gap_in[MAXGAP];
static char irqbuf[2][16384];

static unsigned long long irq_total(const char *s) {
  unsigned long long t = 0;
  const char *line = strchr(s, '\n');
  while (line) {
    line++;
    const char *c = strchr(line, ':'), *nl = strchr(line, '\n');
    if (!nl) nl = line + strlen(line);
    if (c && c < nl) {
      const char *q = c + 1;
      for (;;) {
        while (*q == ' ') q++;
        if (q >= nl || *q < '0' || *q > '9') break;
        unsigned long long v = 0;
        while (*q >= '0' && *q <= '9') v = v * 10 + (unsigned long long)(*q++ - '0');
        t += v;
      }
    }
    line = *nl ? nl : 0;
  }
  return t;
}

static int cmp(const void *a, const void *b) {
  unsigned long long x = *(const unsigned long long *)a, y = *(const unsigned long long *)b;
  return x < y ? -1 : x > y;
}

int main(void) {
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_sigill;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGILL, &sa, 0);
  volatile int stage = 0;
  if (sigsetjmp(jb, 1)) {
    printf("PLAINPROBE access=trapped at=%s\n", stage == 0 ? "rdcycle" : "rdinstret");
    return 2;
  }
  unsigned long long c0 = rdc();
  stage = 1;
  unsigned long long i0 = rdi();
  signal(SIGILL, SIG_DFL);
  unsigned long long c1 = rdc(), i1 = rdi();
  printf("PLAINPROBE access=ok cycle_advances=%d instret_advances=%d\n", c1 > c0, i1 > i0);

  int fd = open("/proc/interrupts", O_RDONLY);
  long n0 = fd >= 0 ? pread(fd, irqbuf[0], sizeof irqbuf[0] - 1, 0) : -1;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  unsigned long long cs = rdc(), is = rdi();
  unsigned long long end = cs + (unsigned long long)SPIN_SECONDS * 25000000ULL; /* refined below by ns */
  unsigned long long prev_c = cs, prev_i = is, gaps = 0, iters = 0;
  for (;;) {
    unsigned long long c = rdc(), i = rdi();
    if (c - prev_c > GAP_CYCLES) {
      if (gaps < MAXGAP) { gap_cy[gaps] = c - prev_c; gap_in[gaps] = i - prev_i; }
      gaps++;
    }
    prev_c = c; prev_i = i; iters++;
    if (c >= end) break;
  }
  unsigned long long ce = rdc(), ie = rdi();
  clock_gettime(CLOCK_MONOTONIC, &t1);
  long n1 = fd >= 0 ? pread(fd, irqbuf[1], sizeof irqbuf[1] - 1, 0) : -1;
  long long ns = (long long)(t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
  unsigned long long irqd = 0;
  if (n0 > 0 && n1 > 0) {
    irqbuf[0][n0] = irqbuf[1][n1] = 0;
    irqd = irq_total(irqbuf[1]) - irq_total(irqbuf[0]);
  }
  printf("PLAINPROBE spin cycles=%llu instret=%llu ns=%lld cycles_per_us=%.3f loop_iters=%llu gaps=%llu "
         "proc_interrupts_delta=%llu%s\n", ce - cs, ie - is, ns, ns > 0 ? (double)(ce - cs) * 1000.0 / (double)ns : 0.0,
         iters, gaps, irqd, fd >= 0 ? "" : " proc_interrupts=absent");
  unsigned long long g = gaps < MAXGAP ? gaps : MAXGAP, sum_c = 0, sum_i = 0;
  for (unsigned long long k = 0; k < g; k++) { sum_c += gap_cy[k]; sum_i += gap_in[k]; }
  if (g) {
    qsort(gap_cy, g, sizeof gap_cy[0], cmp);
    qsort(gap_in, g, sizeof gap_in[0], cmp);
    printf("PLAINPROBE gap_cycles min=%llu p10=%llu median=%llu p90=%llu max=%llu mean=%llu\n", gap_cy[0],
           gap_cy[g / 10], gap_cy[g / 2], gap_cy[g * 9 / 10], gap_cy[g - 1], sum_c / g);
    printf("PLAINPROBE gap_instret min=%llu p10=%llu median=%llu p90=%llu max=%llu mean=%llu\n", gap_in[0],
           gap_in[g / 10], gap_in[g / 2], gap_in[g * 9 / 10], gap_in[g - 1], sum_i / g);
  }
  struct rusage r0, r1;
  char *m = mmap(0, (size_t)FAULT_PAGES * 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m != MAP_FAILED) {
    getrusage(RUSAGE_SELF, &r0);
    for (int k = 0; k < FAULT_PAGES; k++) {
      unsigned long long a = rdc(), b = rdi();
      *(volatile char *)(m + (size_t)k * 4096) = 1;
      unsigned long long b2 = rdi(), a2 = rdc();
      f_cy[k] = a2 - a; f_in[k] = b2 - b;
    }
    getrusage(RUSAGE_SELF, &r1);
    qsort(f_cy, FAULT_PAGES, sizeof f_cy[0], cmp);
    qsort(f_in, FAULT_PAGES, sizeof f_in[0], cmp);
    printf("PLAINPROBE fault pages=%d minflt_delta=%ld cycles_min=%llu cycles_median=%llu cycles_p90=%llu "
           "instret_min=%llu instret_median=%llu instret_p90=%llu\n", FAULT_PAGES, r1.ru_minflt - r0.ru_minflt,
           f_cy[0], f_cy[FAULT_PAGES / 2], f_cy[FAULT_PAGES * 9 / 10], f_in[0], f_in[FAULT_PAGES / 2],
           f_in[FAULT_PAGES * 9 / 10]);
  } else {
    printf("PLAINPROBE fault mmap_failed\n");
  }
  printf("PLAINPROBE done\n");
  return 0;
}
