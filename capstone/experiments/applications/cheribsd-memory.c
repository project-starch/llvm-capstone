/* Measurement-only phase hook for complete CheriBSD applications.
 * Jemalloc's allocation ledger includes MRS quarantine; it is not live payload.
 * Epochs come from the read-only kernel info page; no observer sweep is issued.
 * Emulator cycle fields are deliberately not reported as performance. */
#include <stddef.h>
#include <stdint.h>
#include <cheri/revoke.h>
#include <malloc_np.h>
#include <sys/resource.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int __real_main(int, char **);
ssize_t __real_write(int, const void *, size_t);
static const struct cheri_revoke_info *info;
static int shadow_error;

static size_t counter(const char *name, int *error) {
  size_t value = 0, length = sizeof value;
  int rc = mallctl(name, &value, &length, NULL, 0);
  if (rc && !*error) *error = rc;
  return value;
}

#ifdef EXP_ALLOCATIONS
void exp_alloc_start(void);
void exp_alloc_report(const char *);
#endif

static void report(const char *phase) {
#ifdef EXP_ALLOCATIONS
  exp_alloc_report(phase);
#endif
  struct rusage usage = {0};
  uint64_t epoch = 1;
  size_t length = sizeof epoch;
  int heap_error = mallctl("epoch", &epoch, &length, &epoch, sizeof epoch);
  size_t allocated = counter("stats.allocated", &heap_error);
  size_t active = counter("stats.active", &heap_error);
  size_t resident = counter("stats.resident", &heap_error);
  size_t mapped = counter("stats.mapped", &heap_error);
  getrusage(RUSAGE_SELF, &usage);
  char line[1024];
  int n = snprintf(line, sizeof line,
    "EXP-CHERI phase=%s revocation=%d heap_error=%d shadow_error=%d "
    "allocated=%zu active=%zu resident=%zu mapped=%zu maxrss_kib=%ld "
    "enqueue=%llu dequeue=%llu\n", phase,
    malloc_revoke_enabled(), heap_error, shadow_error,
    allocated, active, resident, mapped, usage.ru_maxrss,
    (unsigned long long)(info ? info->epochs.enqueue : 0),
    (unsigned long long)(info ? info->epochs.dequeue : 0));
  if (n > 0 && (size_t)n < sizeof line) __real_write(2, line, n);
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
  void *shadow = NULL;
  if (cheri_revoke_get_shadow(CHERI_REVOKE_SHADOW_INFO_STRUCT, NULL, &shadow)) shadow_error = errno;
  else info = shadow;
  if (!info && !shadow_error) shadow_error = EFAULT;
  if (atexit(at_exit)) return 125;
#ifdef EXP_ALLOCATIONS
  exp_alloc_start();
#endif
  report("startup");
  return __real_main(argc, argv);
}
