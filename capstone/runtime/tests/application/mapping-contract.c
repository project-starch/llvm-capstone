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
 * never produce the malformed request. */
#define _GNU_SOURCE
#include <errno.h>
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

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "basic";
  int r;
  if (!strcmp(mode, "basic")) r = basic();
  else if (!strcmp(mode, "refresh")) r = refresh();
  else if (!strcmp(mode, "two")) r = two();
  else if (!strcmp(mode, "errors")) r = errors();
  else if (!strcmp(mode, "alias-fault")) r = alias_fault();
  else { fprintf(stderr, "mapping-contract: unknown mode %s\n", mode); return 2; }
  if (r)
    return r;
  printf("mapping-contract %s: PASS\n", mode);
  return 0;
}
