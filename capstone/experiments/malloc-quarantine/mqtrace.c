/* LD_PRELOAD tracer for unmodified programs: sampled allocator state and the
 * reuse histogram, taken from the program's own malloc/free calls.
 *
 * Interposes malloc, calloc, realloc, free, posix_memalign and aligned_alloc
 * and forwards each to the next definition (libc's, i.e. MRS over jemalloc),
 * so the allocator under test is unchanged.  Output, on stderr:
 *   MQ op= live_req= live_asked= allocated= active= resident= mapped= metadata=
 *      maxrss_kib= enqueue= dequeue= fresh_addr= reused_addr= win_pages= win_lines=
 *   MQ-HIST b:count ...   MQ-STRIDE b:count ...   MQ-SIZES b:count ...
 *   MQ-DONE objects= ops= peak_live= peak_live_objects= peak_live_pow2_256=
 * peak_live is exact (updated on every allocation), unlike the sampled lines.
 * With MQ_TRACK=1: MQ-SIZES is the log2 histogram of the asked sizes;
 * peak_live_objects the most objects held at once; peak_live_pow2_256 the
 * peak of the held bytes if every object were rounded up to a power of two
 * of at least 256 bytes, which is what a buddy allocator with 256-byte atoms
 * (Capstone's Sublet heap) would hold for the same objects.
 * op counts allocations (malloc-like calls); a line is printed every
 * MQ_SAMPLE allocations (default 4096).  live_req is the usable size of the
 * objects the program holds; metadata is jemalloc's stats.metadata (its own
 * bookkeeping, included in resident and mapped).  With MQ_TRACK=1 the reuse distance (in
 * allocations between the free of an address and its next allocation) and
 * the distinct 4 KiB pages / 64 B lines handed out per window are recorded,
 * and also:
 *   live_asked  the bytes the program asked for (sum of n over live objects),
 *               so usable - asked is jemalloc's size-class rounding;
 *   MQ-STRIDE   log2 histogram of |address - previous allocation's address|
 *               (bucket 0 = same address), a measure of spatial locality.
 * MQ_WIN=0 skips the per-window page/line sets (their cost grows with object
 * size) and reports win_pages=win_lines=0.
 * MQ_TRACK=2 keeps only the live objects (the table holds an address from its
 * allocation to its free), for programs that hand out more distinct addresses
 * than the reuse table holds: MQ-SIZES, live_asked, peak_live_objects and
 * peak_live_pow2_256 as with MQ_TRACK=1; no reuse distance, stride or windows
 * (fresh_addr, reused_addr, MQ-HIST and MQ-STRIDE stay zero).
 * MQ_TRACK=3 measures the reuse distance on a fixed sample of the addresses:
 * an address is followed when a hash of it falls in 1 of MQ_ADDR_SAMPLE
 * buckets (default 16), and then on every allocation and free, so each
 * followed address contributes all its reuses.  Distances still count all
 * allocations.  MQ-HIST, fresh_addr and reused_addr cover the followed
 * addresses only; nothing else is tracked (no sizes, live objects, stride).
 * The tables are mmap'd, outside the measured heap (but inside max RSS, so
 * footprint runs leave tracking off).
 *
 * MQ_PREFIX replaces the "MQ" of every line, for programs that print MQ lines
 * of their own.
 *
 * Threads: every counter and table is updated under one spin lock, and the
 * guard against the tracer's own allocations (busy) is per thread, so
 * multi-threaded programs (mleak) are counted exactly.
 *
 * Limits: allocations made
 * inside libc without going through its PLT (e.g. reallocarray) are not
 * seen by the tracker, though the sampled jemalloc ledger still counts them.
 */
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <cheri/revoke.h>
#include <malloc_np.h>

static void *(*real_malloc)(size_t);
static void (*real_free)(void *);
static void *(*real_calloc)(size_t, size_t);
static void *(*real_realloc)(void *, size_t);
static int (*real_posix_memalign)(void **, size_t, size_t);
static void *(*real_aligned_alloc)(size_t, size_t);

static __thread int busy __attribute__((tls_model("initial-exec")));
static int track, win, ready, lock_word;
static void lock(void) { while (__atomic_test_and_set(&lock_word, __ATOMIC_ACQUIRE)) ; }
static void unlock(void) { __atomic_clear(&lock_word, __ATOMIC_RELEASE); }
static uint64_t op, every = 4096, live, objects, peak_live;
static const char *pfx = "MQ";

static void resolve(void) {
  real_malloc = dlsym(RTLD_NEXT, "malloc");
  real_free = dlsym(RTLD_NEXT, "free");
  real_calloc = dlsym(RTLD_NEXT, "calloc");
  real_realloc = dlsym(RTLD_NEXT, "realloc");
  real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
  real_aligned_alloc = dlsym(RTLD_NEXT, "aligned_alloc");
  if (!real_malloc || !real_free || !real_calloc || !real_realloc) {
    write(2, "MQ-ERROR dlsym\n", 15);
    _exit(6);
  }
}

static void *table(size_t bytes) {
  void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (p == MAP_FAILED) { write(2, "MQ-ERROR mmap\n", 14); _exit(4); }
  return p;
}

/* Address -> op of its last free (0 = live).  Open addressing. */
#define TBITS 24
#define TSIZE (1u << TBITS)
static uint64_t *tkey, *tfree, *treq, seen, fresh, reused, hist[64];
static uint64_t asked, prev_addr, stride[65], sizes[65];
static uint64_t live_objs, peak_live_objs, live_pow2, peak_live_pow2;
static uint64_t pow2_256(uint64_t n) {
  uint64_t b = 256;
  while (b < n) b <<= 1;
  return b;
}
static void on_alloc(uint64_t a, uint64_t n) {
  uint64_t d = a > prev_addr ? a - prev_addr : prev_addr - a;
  stride[d ? 64 - __builtin_clzll(d) : 0]++;
  prev_addr = a;
  asked += n;
  sizes[n ? 64 - __builtin_clzll(n) : 0]++;
  live_objs++; if (live_objs > peak_live_objs) peak_live_objs = live_objs;
  live_pow2 += pow2_256(n); if (live_pow2 > peak_live_pow2) peak_live_pow2 = live_pow2;
  uint64_t h = (a >> 4) * 0x9e3779b97f4a7c15ull >> (64 - TBITS);
  for (;;) {
    if (tkey[h] == a) {
      if (tfree[h]) {
        uint64_t r = op - tfree[h];
        hist[r ? 64 - __builtin_clzll(r) : 0]++;
        reused++; tfree[h] = 0;
      }
      treq[h] = n;
      return;
    }
    if (tkey[h] == 0) {
      if (++seen > TSIZE / 2) { write(2, "MQ-ERROR table full\n", 20); _exit(3); }
      tkey[h] = a; tfree[h] = 0; treq[h] = n; fresh++; return;
    }
    h = (h + 1) & (TSIZE - 1);
  }
}
/* MQ_TRACK=2: the same key/size tables, holding live objects only.  Linear
 * probing with backward-shift deletion, so no tombstones accumulate. */
static uint64_t slot_of(uint64_t a) { return (a >> 4) * 0x9e3779b97f4a7c15ull >> (64 - TBITS); }
static void live_alloc(uint64_t a, uint64_t n) {
  asked += n;
  sizes[n ? 64 - __builtin_clzll(n) : 0]++;
  uint64_t h = slot_of(a);
  for (;;) {
    if (tkey[h] == a) { /* still recorded live: replace, as on_alloc does */
      asked = asked >= treq[h] ? asked - treq[h] : 0;
      { uint64_t r = pow2_256(treq[h]); live_pow2 = live_pow2 >= r ? live_pow2 - r : 0; }
      if (live_objs) live_objs--;
      break;
    }
    if (tkey[h] == 0) {
      if (++seen > TSIZE / 2) { write(2, "MQ-ERROR table full\n", 20); _exit(3); }
      tkey[h] = a;
      break;
    }
    h = (h + 1) & (TSIZE - 1);
  }
  treq[h] = n;
  live_objs++; if (live_objs > peak_live_objs) peak_live_objs = live_objs;
  live_pow2 += pow2_256(n); if (live_pow2 > peak_live_pow2) peak_live_pow2 = live_pow2;
}
static void live_free(uint64_t a) {
  uint64_t i = slot_of(a);
  for (;;) {
    if (tkey[i] == 0) return; /* allocated before tracking began */
    if (tkey[i] == a) break;
    i = (i + 1) & (TSIZE - 1);
  }
  asked = asked >= treq[i] ? asked - treq[i] : 0;
  if (live_objs) live_objs--;
  { uint64_t r = pow2_256(treq[i]); live_pow2 = live_pow2 >= r ? live_pow2 - r : 0; }
  seen--;
  for (;;) { /* backward shift: pull later entries of the run into the hole */
    tkey[i] = 0; treq[i] = 0;
    uint64_t j = i;
    for (;;) {
      j = (j + 1) & (TSIZE - 1);
      if (tkey[j] == 0) return;
      uint64_t k = slot_of(tkey[j]);
      if (i <= j ? (i < k && k <= j) : (i < k || k <= j)) continue; /* already reachable */
      break;
    }
    tkey[i] = tkey[j]; treq[i] = treq[j];
    i = j;
  }
}

/* MQ_TRACK=3: reuse distance on a hash-selected sample of the addresses. */
static uint64_t addr_keep = 16;
static int followed(uint64_t a) {
  return ((a >> 4) * 0xd6e8feb86659fd93ull >> 32) % addr_keep == 0;
}
static void sampled_alloc(uint64_t a) {
  uint64_t h = slot_of(a);
  for (;;) {
    if (tkey[h] == a) {
      if (tfree[h]) {
        uint64_t r = op - tfree[h];
        hist[r ? 64 - __builtin_clzll(r) : 0]++;
        reused++; tfree[h] = 0;
      }
      return;
    }
    if (tkey[h] == 0) {
      if (++seen > TSIZE / 2) { write(2, "MQ-ERROR table full\n", 20); _exit(3); }
      tkey[h] = a; tfree[h] = 0; fresh++; return;
    }
    h = (h + 1) & (TSIZE - 1);
  }
}
static void sampled_free(uint64_t a) {
  uint64_t h = slot_of(a);
  for (;;) {
    if (tkey[h] == a) { tfree[h] = op ? op : 1; return; }
    if (tkey[h] == 0) return; /* allocated before tracking began */
    h = (h + 1) & (TSIZE - 1);
  }
}

static void on_free(uint64_t a) {
  uint64_t h = (a >> 4) * 0x9e3779b97f4a7c15ull >> (64 - TBITS);
  for (;;) {
    if (tkey[h] == a) {
      tfree[h] = op ? op : 1;
      asked = asked >= treq[h] ? asked - treq[h] : 0;
      if (live_objs) live_objs--;
      { uint64_t r = pow2_256(treq[h]); live_pow2 = live_pow2 >= r ? live_pow2 - r : 0; }
      treq[h] = 0;
      return;
    }
    if (tkey[h] == 0) return; /* allocated before tracking began */
    h = (h + 1) & (TSIZE - 1);
  }
}

#define WBITS 20
static uint64_t *wpage, *wline, *wpage_s, *wline_s, wpages, wlines, wstamp = 1;
static void wadd(uint64_t *k, uint64_t *s, uint64_t v, uint64_t *count) {
  uint64_t h = v * 0x9e3779b97f4a7c15ull >> (64 - WBITS);
  for (;;) {
    if (s[h] != wstamp) { s[h] = wstamp; k[h] = v; (*count)++; return; }
    if (k[h] == v) return;
    h = (h + 1) & ((1u << WBITS) - 1);
  }
}

static void sample(void) {
  size_t allocated = 0, active = 0, resident = 0, mapped = 0, metadata = 0, len;
  uint64_t e = 1; len = sizeof e;
  mallctl("epoch", &e, &len, &e, sizeof e);
  len = sizeof(size_t); mallctl("stats.allocated", &allocated, &len, NULL, 0);
  len = sizeof(size_t); mallctl("stats.active", &active, &len, NULL, 0);
  len = sizeof(size_t); mallctl("stats.resident", &resident, &len, NULL, 0);
  len = sizeof(size_t); mallctl("stats.mapped", &mapped, &len, NULL, 0);
  len = sizeof(size_t); mallctl("stats.metadata", &metadata, &len, NULL, 0);
  static const struct cheri_revoke_info *info;
  if (!info) { void *p = NULL;
    if (!cheri_revoke_get_shadow(CHERI_REVOKE_SHADOW_INFO_STRUCT, NULL, &p)) info = p; }
  struct rusage ru; getrusage(RUSAGE_SELF, &ru);
  char line[512];
  int n = snprintf(line, sizeof line,
    "%s op=%llu live_req=%llu live_asked=%llu allocated=%zu active=%zu resident=%zu mapped=%zu "
    "metadata=%zu maxrss_kib=%ld enqueue=%llu dequeue=%llu fresh_addr=%llu reused_addr=%llu "
    "win_pages=%llu win_lines=%llu\n", pfx,
    (unsigned long long)op, (unsigned long long)live, (unsigned long long)asked, allocated, active, resident, mapped,
    metadata, ru.ru_maxrss, info ? (unsigned long long)info->epochs.enqueue : 0ull,
    info ? (unsigned long long)info->epochs.dequeue : 0ull, (unsigned long long)fresh,
    (unsigned long long)reused, (unsigned long long)wpages, (unsigned long long)wlines);
  if (n > 0) write(2, line, (size_t)n);
  wstamp++; wpages = wlines = 0;
}

__attribute__((constructor)) static void mq_init(void) {
  if (!real_malloc) resolve();
  busy++;
  const char *s = getenv("MQ_SAMPLE"), *t = getenv("MQ_TRACK"), *x = getenv("MQ_PREFIX"),
             *w = getenv("MQ_WIN");
  if (x && *x && strlen(x) < 8) pfx = x;
  if (s && atoll(s) > 0) every = (uint64_t)atoll(s);
  if (t && *t == '3') {
    const char *k = getenv("MQ_ADDR_SAMPLE");
    if (k && atoll(k) > 0) addr_keep = (uint64_t)atoll(k);
    tkey = table(TSIZE * 8ul); tfree = table(TSIZE * 8ul);
    track = 3;
  } else if (t && *t == '2') {
    tkey = table(TSIZE * 8ul); treq = table(TSIZE * 8ul);
    track = 2;
  } else if (t && *t == '1') {
    tkey = table(TSIZE * 8ul); tfree = table(TSIZE * 8ul); treq = table(TSIZE * 8ul);
    wpage = table(8ul << WBITS); wline = table(8ul << WBITS);
    wpage_s = table(8ul << WBITS); wline_s = table(8ul << WBITS);
    track = 1;
    win = !(w && *w == '0');
  }
  ready = 1;
  sample();
  busy--;
}

static void got(void *p, size_t n) {
  if (!p || !ready || busy) return;
  busy++;
  lock();
  op++; objects++;
  size_t sz = malloc_usable_size(p);
  live += sz;
  if (live > peak_live) peak_live = live;
  if (track == 3) { uint64_t a = (uint64_t)(uintptr_t)p; if (followed(a)) sampled_alloc(a); }
  else if (track == 2) live_alloc((uint64_t)(uintptr_t)p, n);
  else if (track) {
    uint64_t a = (uint64_t)(uintptr_t)p;
    on_alloc(a, n);
    for (uint64_t b = a & ~63ull; win && b < a + sz; b += 64) {
      wadd(wline, wline_s, b >> 6, &wlines);
      wadd(wpage, wpage_s, b >> 12, &wpages);
    }
  }
  if (op % every == 0) sample();
  unlock();
  busy--;
}

static void gone(void *p) {
  if (!p || !ready || busy) return;
  busy++;
  lock();
  size_t sz = malloc_usable_size(p);
  live = live >= sz ? live - sz : 0;
  if (track == 3) { uint64_t a = (uint64_t)(uintptr_t)p; if (followed(a)) sampled_free(a); }
  else if (track == 2) live_free((uint64_t)(uintptr_t)p);
  else if (track) on_free((uint64_t)(uintptr_t)p);
  unlock();
  busy--;
}

void *malloc(size_t n) {
  if (!real_malloc) resolve();
  void *p = real_malloc(n); got(p, n); return p;
}
void *calloc(size_t a, size_t b) {
  if (!real_calloc) resolve();
  void *p = real_calloc(a, b); got(p, a * b); return p;
}
void free(void *p) {
  if (!real_free) resolve();
  gone(p); real_free(p);
}
void *realloc(void *old, size_t n) {
  if (!real_realloc) resolve();
  gone(old);
  void *p = real_realloc(old, n);
  if (p) got(p, n);
  else if (old && n) got(old, n); /* failed: the old object is still live (its size is approximated by n) */
  return p;
}
int posix_memalign(void **out, size_t al, size_t n) {
  if (!real_posix_memalign) resolve();
  int r = real_posix_memalign(out, al, n);
  if (r == 0) got(*out, n);
  return r;
}
void *aligned_alloc(size_t al, size_t n) {
  if (!real_aligned_alloc) resolve();
  void *p = real_aligned_alloc(al, n); got(p, n); return p;
}

__attribute__((destructor)) static void mq_fini(void) {
  if (!ready) return;
  busy++;
  lock();
  sample();
  char line[4096];
  int n = snprintf(line, sizeof line, "%s-HIST", pfx);
  for (int b = 0; b < 64 && n > 0 && (size_t)n < sizeof line - 32; b++)
    if (hist[b]) n += snprintf(line + n, sizeof line - n, " %d:%llu", b, (unsigned long long)hist[b]);
  n += snprintf(line + n, sizeof line - n, "\n%s-STRIDE", pfx);
  for (int b = 0; b < 65 && n > 0 && (size_t)n < sizeof line - 64; b++)
    if (stride[b]) n += snprintf(line + n, sizeof line - n, " %d:%llu", b, (unsigned long long)stride[b]);
  n += snprintf(line + n, sizeof line - n, "\n%s-SIZES", pfx);
  for (int b = 0; b < 65 && n > 0 && (size_t)n < sizeof line - 96; b++)
    if (sizes[b]) n += snprintf(line + n, sizeof line - n, " %d:%llu", b, (unsigned long long)sizes[b]);
  n += snprintf(line + n, sizeof line - n, "\n%s-DONE objects=%llu ops=%llu peak_live=%llu "
                "peak_live_objects=%llu peak_live_pow2_256=%llu\n", pfx,
                (unsigned long long)objects, (unsigned long long)op, (unsigned long long)peak_live,
                (unsigned long long)peak_live_objs, (unsigned long long)peak_live_pow2);
  write(2, line, (size_t)n);
  unlock();
  busy--;
}
