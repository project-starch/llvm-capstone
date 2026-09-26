/* Link with --wrap=main,--wrap=write. Measurement builds only.
 * Workloads issue a single write(2, "MEMPHASE <name>\n", length) at a phase
 * boundary. Report before forwarding it, without allocating or reading
 * application pointers beyond that write's bounds. The exit sample also
 * covers applications that call exit instead of returning from main. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int __real_main(int, char **);
ssize_t __real_write(int, const void *, size_t);

#ifdef EXP_SUBLET
size_t __capstone_sublet_live_bytes(void);
size_t __capstone_sublet_peak_bytes(void);
size_t __capstone_sublet_pool_bytes(void);
size_t __capstone_sublet_table_bytes(void);
void __capstone_sublet_heap_stats(unsigned long [9]);
#else
size_t __capstone_level0_in_use(void);
size_t __capstone_level0_peak_in_use(void);
size_t __capstone_level0_peak_end(void);
size_t __capstone_level0_arena_bytes(void);
#endif

static void report(const char *phase) {
  char line[640];
#ifdef EXP_SUBLET
  unsigned long c[9];
  __capstone_sublet_heap_stats(c);
  int n = snprintf(line, sizeof line, "EXP-MEM phase=%s", phase);
#define FIELD(name, value) do { \
  n += snprintf(line+n, sizeof line-(size_t)n, " " name "=%lu", (unsigned long)(value)); \
} while (0)
  FIELD("live", __capstone_sublet_live_bytes());
  FIELD("peak", __capstone_sublet_peak_bytes());
  FIELD("pool", __capstone_sublet_pool_bytes());
  FIELD("tables", __capstone_sublet_table_bytes());
  FIELD("allocs", c[0]); FIELD("frees", c[1]); FIELD("merges", c[2]);
  FIELD("split", c[4]); FIELD("mrev", c[5]); FIELD("revoke", c[7]); FIELD("init", c[8]);
  line[n++] = '\n';
#undef FIELD
#else
  int n = snprintf(line, sizeof line,
      "EXP-MEM phase=%s live=%zu peak=%zu end=%zu pool=%zu\n", phase,
      __capstone_level0_in_use(), __capstone_level0_peak_in_use(),
      __capstone_level0_peak_end(), __capstone_level0_arena_bytes());
#endif
  if (n > 0 && (size_t)n < sizeof line)
    __real_write(2, line, (size_t)n);
}

ssize_t __wrap_write(int fd, const void *buf, size_t n) {
  if (fd == 2 && n > 10 && n < 80 && !memcmp(buf, "MEMPHASE ", 9)) {
    char phase[80];
    memcpy(phase, (const char *)buf + 9, n - 9);
    phase[n - 9] = 0;
    for (size_t i = 0; phase[i]; ++i)
      if (phase[i] == '\n' || phase[i] == '\r' || phase[i] == ' ') phase[i] = 0;
    report(phase);
  }
  return __real_write(fd, buf, n);
}

static void at_exit(void) { report("exit"); }
int __wrap_main(int argc, char **argv) {
#ifdef EXP_PYMALLOC
  extern int cpy_sublet_init(void);
  if (cpy_sublet_init() < 0) return 124;
#endif
  if (atexit(at_exit)) return 125;
  report("startup");
  return __real_main(argc, argv);
}
