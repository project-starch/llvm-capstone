/* Online application allocation accounting, single-thread measurement builds.
 * Store integer addresses only: the observer never retains heap capabilities.
 * No event stream, observer allocation, policy change, or forced collection.
 * --wrap covers linked application/static-library calls, not dynamic libc's
 * private allocations. Nested allocator calls are counted at their outer API.
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef EXP_ADDRESS_BITS
#define EXP_ADDRESS_BITS 16
#endif
#define CAPACITY (1UL << EXP_ADDRESS_BITS)
struct entry { uint64_t address, size_plus_one, freed_at; };
static struct entry entries[CAPACITY];
static uint64_t operations, allocations, frees, failures, unique, reused;
static uint64_t live, peak, blocks, peak_blocks, requested, unknown, errors;
static uint64_t reuse_distance[6];
static int enabled, busy;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);
ssize_t __real_write(int, const void *, size_t);

static uint64_t address(const void *p) {
#ifdef EXP_CAPSTONE
  return __builtin_capstone_cap_get_cursor((void *)p);
#elif defined(__CHERI_PURE_CAPABILITY__)
  return __builtin_cheri_address_get(p);
#else
  return (uintptr_t)p;
#endif
}

static struct entry *lookup(uint64_t addr, int create) {
  if (!addr) return NULL;
  size_t index = ((addr >> 4) * UINT64_C(11400714819323198485)) >>
                 (64 - EXP_ADDRESS_BITS);
  for (size_t i = 0; i < CAPACITY; ++i) {
    struct entry *e = &entries[(index + i) & (CAPACITY - 1)];
    if (e->address == addr) return e;
    if (!e->address) {
      if (!create) return NULL;
      e->address = addr;
      ++unique;
      return e;
    }
  }
  ++errors; /* Never silently evict history and inflate the reuse rate. */
  return NULL;
}

static void remove_address(uint64_t addr) {
  if (!addr) return;
  struct entry *e = lookup(addr, 0);
  if (!e || !e->size_plus_one) { ++unknown; return; }
  live -= e->size_plus_one - 1;
  --blocks;
  ++frees;
  e->size_plus_one = 0;
  e->freed_at = operations;
}

static void add_address(void *p, size_t n, int continuation) {
  if (!p) { ++failures; return; }
  ++allocations;
  requested += n;
  struct entry *e = lookup(address(p), 1);
  if (!e) return;
  if (e->size_plus_one || n == UINT64_MAX) { ++errors; return; }
  if (e->freed_at && !continuation) {
    ++reused;
    uint64_t distance = operations - e->freed_at;
    size_t bin = distance <= 1 ? 0 : distance <= 8 ? 1 : distance <= 64 ? 2 :
                 distance <= 512 ? 3 : distance <= 4096 ? 4 : 5;
    ++reuse_distance[bin];
  }
  e->size_plus_one = n + 1;
  live += n;
  ++blocks;
  if (live > peak) peak = live;
  if (blocks > peak_blocks) peak_blocks = blocks;
}

void *__wrap_malloc(size_t n) {
  if (!enabled || busy) return __real_malloc(n);
  busy = 1; ++operations;
  void *p = __real_malloc(n);
  add_address(p, n, 0); busy = 0;
  return p;
}
void *__wrap_calloc(size_t n, size_t size) {
  if (!enabled || busy) return __real_calloc(n, size);
  busy = 1; ++operations;
  void *p = __real_calloc(n, size);
  if (size && n > SIZE_MAX / size) {
    if (p) ++errors; else ++failures;
  } else add_address(p, n * size, 0);
  busy = 0;
  return p;
}
void __wrap_free(void *p) {
  if (!enabled || busy) { __real_free(p); return; }
  busy = 1;
  remove_address(address(p));
  __real_free(p); busy = 0;
}
void *__wrap_realloc(void *p, size_t n) {
  if (!enabled || busy) return __real_realloc(p, n);
  busy = 1; ++operations;
  uint64_t old = address(p); /* Save before realloc can invalidate p. */
  void *q = __real_realloc(p, n);
  /* Supported libc targets free p for realloc(p, 0). A failed nonzero
   * realloc preserves the old allocation and its accounting. */
  if (q || !n) remove_address(old);
  if (q) add_address(q, n, old && old == address(q));
  else if (n) ++failures;
  busy = 0;
  return q;
}

#ifndef EXP_CAPSTONE
int __real_posix_memalign(void **, size_t, size_t);
void *__real_aligned_alloc(size_t, size_t);
int __wrap_posix_memalign(void **out, size_t alignment, size_t n) {
  if (!enabled || busy) return __real_posix_memalign(out, alignment, n);
  busy = 1; ++operations;
  int result = __real_posix_memalign(out, alignment, n);
  if (result) ++failures; else add_address(*out, n, 0);
  busy = 0;
  return result;
}
void *__wrap_aligned_alloc(size_t alignment, size_t n) {
  if (!enabled || busy) return __real_aligned_alloc(alignment, n);
  busy = 1; ++operations;
  void *p = __real_aligned_alloc(alignment, n);
  add_address(p, n, 0); busy = 0;
  return p;
}
#endif

void exp_alloc_start(void) { enabled = 1; }
void exp_alloc_report(const char *phase) {
  int saved_errno = errno;
  busy = 1;
  uint64_t low = UINT64_MAX, high = 0;
  for (size_t i = 0; i < CAPACITY; ++i) {
    struct entry *e = &entries[i];
    if (!e->size_plus_one) continue;
    if (e->address < low) low = e->address;
    uint64_t end = e->address + e->size_plus_one - 1;
    if (end > high) high = end;
  }
  char line[1024];
  int n = snprintf(line, sizeof line, "EXP-ALLOC phase=%s", phase);
#define FIELD(name, value) do { \
  if (n >= 0 && (size_t)n < sizeof line) \
    n += snprintf(line+n, sizeof line-(size_t)n, " " name "=%llu", \
                  (unsigned long long)(value)); \
} while (0)
  FIELD("live", live);
  FIELD("peak", peak);
  FIELD("blocks", blocks);
  FIELD("peak_blocks", peak_blocks);
  FIELD("requested", requested);
  FIELD("allocations", allocations);
  FIELD("frees", frees);
  FIELD("failures", failures);
  FIELD("unique", unique);
  FIELD("reused", reused);
  FIELD("unknown", unknown);
  FIELD("errors", errors);
  FIELD("span", blocks ? high-low : 0);
  FIELD("observer", sizeof entries);
  FIELD("reuse1", reuse_distance[0]);
  FIELD("reuse8", reuse_distance[1]);
  FIELD("reuse64", reuse_distance[2]);
  FIELD("reuse512", reuse_distance[3]);
  FIELD("reuse4096", reuse_distance[4]);
  FIELD("reuse_more", reuse_distance[5]);
#undef FIELD
  if (n >= 0 && (size_t)n < sizeof line) line[n++] = '\n';
  if (n > 0 && (size_t)n < sizeof line) __real_write(2, line, n);
  busy = 0;
  errno = saved_errno;
}
