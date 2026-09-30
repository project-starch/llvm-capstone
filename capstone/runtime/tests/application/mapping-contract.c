/* The translated-mapping contract of the delegated runtime
 * (docs/plans/mapping-transport-m2.md, section 4, step 6): one mode per
 * promise. Every mode prints "mapping-contract <mode>: PASS" and exits 0, or
 * fails the CHECK that names the broken promise. The alias-fault mode ends in
 * the fault the runtime reports (the launcher exits with SIGSEGV), which is
 * its PASS.
 *
 * mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
 * is a GRANT row; the pointer it returns names memory the domain reaches only
 * through its own capability. The raw rows are used where the libc would
 * never produce the malformed request.
 *
 * Some modes run under capstone-exec-adversary, a launcher that lies
 * (CAPSTONE_ADVERSARY, see runtime/linux/exec.c): alias-fault must still fault
 * when RELEASE is only claimed (release-noop), ro-refused must see EIO when a
 * read-only request is widened (widen), signal must keep its mapping when a
 * signal lands between grant and resume (signal). run-mapping-gate.sh has the
 * pairing of mode, launcher and expected outcome. */
#define _GNU_SOURCE
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(test) do { if (!(test)) { \
  fprintf(stderr, "mapping-contract:%d: %s (errno %d)\n", __LINE__, #test, errno); return 1; \
} } while (0)

#define PAGE 4096UL
#define NR_MAP_GRANT UINT64_C(0xC0DE0008)
#define NR_MAP_RELEASE UINT64_C(0xC0DE0009)

extern long __capstone_delegate_ints(uint64_t nr, uint64_t a, uint64_t b, uint64_t c);

static volatile unsigned char *acquire(size_t len) {
  void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return p == MAP_FAILED ? NULL : p;
}

static int all_zero(volatile unsigned char *p, size_t len) {
  for (size_t i = 0; i < len; ++i)
    if (p[i])
      return 0;
  return 1;
}

/* Acquire, see zero fill, write both endpoints and one word per page, read
 * everything back, release. */
static int basic(void) {
  size_t len = 8 * PAGE;
  volatile unsigned char *p = acquire(len);
  CHECK(p);
  CHECK(all_zero(p, len));
  p[0] = 0x11;
  p[len - 1] = 0x22;
  for (size_t page = 0; page < len / PAGE; ++page)
    *(volatile uint64_t *)(p + page * PAGE + 8) = 0x1000 + page;
  CHECK(p[0] == 0x11 && p[len - 1] == 0x22);
  for (size_t page = 0; page < len / PAGE; ++page)
    CHECK(*(volatile uint64_t *)(p + page * PAGE + 8) == 0x1000 + page);
  for (size_t i = 1; i < len - 1; ++i)
    if (i % PAGE < 8 || i % PAGE >= 16)
      CHECK(p[i] == 0);
  CHECK(!munmap((void *)p, len));
  return 0;
}

/* A released mapping's storage comes back scrubbed: acquire, dirty every
 * byte, release, acquire the same size, see zero fill. */
static int refresh(void) {
  size_t len = 4 * PAGE;
  volatile unsigned char *p = acquire(len);
  CHECK(p);
  for (size_t i = 0; i < len; ++i)
    p[i] = 0xA5;
  CHECK(!munmap((void *)p, len));
  p = acquire(len);
  CHECK(p);
  CHECK(all_zero(p, len));
  CHECK(!munmap((void *)p, len));
  return 0;
}

/* A second live mapping survives the first's release, in both directions. */
static int two(void) {
  volatile unsigned char *a = acquire(2 * PAGE), *b = acquire(3 * PAGE);
  CHECK(a && b);
  a[0] = 1; a[2 * PAGE - 1] = 2;
  b[0] = 3; b[3 * PAGE - 1] = 4;
  CHECK(!munmap((void *)a, 2 * PAGE));
  CHECK(b[0] == 3 && b[3 * PAGE - 1] == 4);
  b[PAGE] = 5;
  CHECK(b[PAGE] == 5);
  volatile unsigned char *c = acquire(PAGE);
  CHECK(c && all_zero(c, PAGE));
  CHECK(b[0] == 3 && b[3 * PAGE - 1] == 4 && b[PAGE] == 5);
  CHECK(!munmap((void *)b, 3 * PAGE));
  CHECK(!munmap((void *)c, PAGE));
  return 0;
}

/* The refusals: a zero length, an unaligned length and an unknown
 * protection at the row, a partial munmap, release of an unknown binding and
 * release twice; then exhaustion of the mapping table and recovery from it. */
static int errors(void) {
  long r;
  CHECK(mmap(NULL, 0, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) == MAP_FAILED &&
        errno == EINVAL);
  CHECK(__capstone_delegate_ints(NR_MAP_GRANT, 0, 6, 0) == -EINVAL);
  CHECK(__capstone_delegate_ints(NR_MAP_GRANT, PAGE - 1, 6, 0) == -EINVAL);
  CHECK(__capstone_delegate_ints(NR_MAP_GRANT, PAGE, 5, 0) == -EINVAL);
  CHECK(__capstone_delegate_ints(NR_MAP_GRANT, PAGE, 0, 0) == -EINVAL);
  CHECK(__capstone_delegate_ints(NR_MAP_RELEASE, 0x7fff, 0, 0) == -ENOENT);
  volatile unsigned char *p = acquire(2 * PAGE);
  CHECK(p);
  CHECK(munmap((void *)p, PAGE) == -1 && errno == EINVAL);
  p[PAGE] = 7;
  CHECK(p[PAGE] == 7);
  CHECK(!munmap((void *)p, 2 * PAGE));
  r = __capstone_delegate_ints(NR_MAP_GRANT, PAGE, 6, 0);
  CHECK(r >= 0);
  CHECK(__capstone_delegate_ints(NR_MAP_RELEASE, (uint64_t)r, 0, 0) == 0);
  CHECK(__capstone_delegate_ints(NR_MAP_RELEASE, (uint64_t)r, 0, 0) == -ENOENT);
  /* exhaustion: whichever table fills first, the failure is ENOMEM or
     ENOSPC, every earlier mapping stays usable, and releasing them all
     restores the service */
  volatile unsigned char *live[64];
  unsigned n = 0;
  for (; n < 64; ++n) {
    live[n] = acquire(PAGE);
    if (!live[n])
      break;
    live[n][0] = (unsigned char)(n + 1);
  }
  CHECK(n >= 16 && n < 64);
  CHECK(errno == ENOMEM || errno == ENOSPC);
  printf("mapping-contract errors: %u mappings before exhaustion (errno %d)\n", n, errno);
  for (unsigned i = 0; i < n; ++i)
    CHECK(live[i][0] == (unsigned char)(i + 1));
  for (unsigned i = 0; i < n; ++i)
    CHECK(!munmap((void *)live[i], PAGE));
  p = acquire(PAGE);
  CHECK(p && all_zero(p, PAGE));
  CHECK(!munmap((void *)p, PAGE));
  return 0;
}

/* A saved alias of a released mapping faults on use: the launcher reports a
 * domain fault and ends with SIGSEGV. Reaching the final line is the failure. */
static int alias_fault(void) {
  volatile unsigned char *p = acquire(PAGE);
  CHECK(p);
  volatile unsigned char *alias = p;
  p[0] = 9;
  CHECK(alias[0] == 9);
  CHECK(!munmap((void *)p, PAGE));
  printf("mapping-contract alias-fault: touching the alias\n");
  fflush(stdout);
  unsigned char v = alias[0];
  fprintf(stderr, "mapping-contract alias-fault: FAIL, the alias read %u\n", v);
  return 1;
}

/* A read-only mapping reads zero and refuses a store: the domain faults
 * (cause 27) and the launcher exits with SIGSEGV. Reaching the end fails. */
static int ro_store(void) {
  void *m = mmap(NULL, PAGE, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  CHECK(m != MAP_FAILED);
  volatile unsigned char *p = m;
  CHECK(p[0] == 0 && p[PAGE - 1] == 0);
  printf("mapping-contract ro-store: storing\n");
  fflush(stdout);
  p[0] = 73;
  fprintf(stderr, "mapping-contract ro-store: FAIL, the store succeeded (%u)\n", p[0]);
  return 1;
}

/* Under the widening launcher: the delivered capability is read-write where
 * read-only was asked for, and the libc refuses it. */
static int ro_refused(void) {
  void *m = mmap(NULL, PAGE, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  CHECK(m == MAP_FAILED);
  CHECK(errno == EIO);
  return 0;
}

/* Release returns the backing: with at most one mapping live, far more
 * grants succeed than the process could hold at once. */
static int cycle(void) {
  for (unsigned i = 0; i < 300; ++i) {
    volatile unsigned char *p = acquire(PAGE);
    if (!p) {
      fprintf(stderr, "mapping-contract cycle: 4 KiB grant %u failed (errno %d)\n", i, errno);
      return 1;
    }
    CHECK(p[0] == 0 && p[PAGE - 1] == 0);
    p[0] = (unsigned char)i;
    CHECK(!munmap((void *)p, PAGE));
  }
  for (unsigned i = 0; i < 60; ++i) {
    volatile unsigned char *p = acquire(16 * PAGE);
    if (!p) {
      fprintf(stderr, "mapping-contract cycle: 64 KiB grant %u failed (errno %d)\n", i, errno);
      return 1;
    }
    CHECK(all_zero(p, 16 * PAGE));
    p[15 * PAGE] = 1;
    CHECK(!munmap((void *)p, 16 * PAGE));
  }
  return 0;
}

/* 2 MiB: two leaf tables. */
static int large(void) {
  size_t len = 512 * PAGE;
  volatile unsigned char *p = acquire(len);
  CHECK(p);
  for (size_t i = 0; i < len; i += PAGE) {
    CHECK(p[i] == 0 && p[i + PAGE - 1] == 0);
    p[i] = (unsigned char)(i / PAGE % 251);
  }
  for (size_t i = 0; i < len; i += PAGE)
    CHECK(p[i] == (unsigned char)(i / PAGE % 251));
  CHECK(!munmap((void *)p, len));
  return 0;
}

/* A signal handled while the grant is being delivered: the handler makes a
 * syscall of its own, and its acknowledgement is another round. The mapping
 * survives. argv[2] says whether the launcher injects the signal. */
static volatile sig_atomic_t seen;
static void note_signal(int sig) {
  (void)sig;
  seen = 1;
  (void)write(1, "mapping-contract signal: handler ran\n", 37);
}
static int signal_mode(int expect_seen) {
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_handler = note_signal;
  sigemptyset(&sa.sa_mask);
  CHECK(!sigaction(SIGUSR1, &sa, NULL));
  volatile unsigned char *p = acquire(PAGE);
  CHECK(p);
  CHECK(seen == expect_seen);
  CHECK(p[0] == 0);
  p[0] = 5;
  CHECK(p[0] == 5);
  CHECK(!munmap((void *)p, PAGE));
  return 0;
}

/* Node pressure (the gate boots with few revocation nodes for this mode): a
 * 2 MiB grant that does not fit is refused with an error, the monitor keeps
 * running, and small grants work before and after. */
static int budget(void) {
  volatile unsigned char *p = acquire(PAGE);
  CHECK(p);
  CHECK(!munmap((void *)p, PAGE));
  p = acquire(512 * PAGE);
  if (p) {
    printf("mapping-contract budget: large=granted\n");
    CHECK(!munmap((void *)p, 512 * PAGE));
  } else {
    printf("mapping-contract budget: large=refused errno=%d\n", errno);
    CHECK(errno == ENOSPC || errno == ENOMEM);
  }
  p = acquire(PAGE);
  CHECK(p && p[0] == 0);
  CHECK(!munmap((void *)p, PAGE));
  return 0;
}

/* Compute without rounds long enough to be preempted, then use a mapping. */
static int spin(void) {
  long start = 0;
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  start = (long)t.tv_sec * 1000 + t.tv_nsec / 1000000;
  for (;;) {
    clock_gettime(CLOCK_MONOTONIC, &t);
    if ((long)t.tv_sec * 1000 + t.tv_nsec / 1000000 - start > 3000)
      break;
  }
  return basic();
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "basic";
  int r;
  if (!strcmp(mode, "basic")) r = basic();
  else if (!strcmp(mode, "refresh")) r = refresh();
  else if (!strcmp(mode, "two")) r = two();
  else if (!strcmp(mode, "errors")) r = errors();
  else if (!strcmp(mode, "alias-fault")) r = alias_fault();
  else if (!strcmp(mode, "ro-store")) r = ro_store();
  else if (!strcmp(mode, "ro-refused")) r = ro_refused();
  else if (!strcmp(mode, "cycle")) r = cycle();
  else if (!strcmp(mode, "large")) r = large();
  else if (!strcmp(mode, "signal")) r = signal_mode(argc > 2 && !strcmp(argv[2], "1"));
  else if (!strcmp(mode, "budget")) r = budget();
  else if (!strcmp(mode, "spin")) r = spin();
  else { fprintf(stderr, "mapping-contract: unknown mode %s\n", mode); return 2; }
  if (r)
    return r;
  printf("mapping-contract %s: PASS\n", mode);
  return 0;
}
