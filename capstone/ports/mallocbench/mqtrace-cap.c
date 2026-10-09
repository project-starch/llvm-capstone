/* Link-time tracer for the virtual Capstone runtime: the counterpart of the
 * CheriBSD side's LD_PRELOAD tracer (llvm-capstone experiments/cheribsd-malloc-quarantine,
 * capstone/experiments/malloc-quarantine/mqtrace.c, MQ_TRACK=1), with the same line format
 * where a field has a meaning on both systems.
 *
 * Linked with -Wl,--wrap=<f> for malloc, calloc, realloc, free, posix_memalign and
 * aligned_alloc. Each wrapper forwards to the runtime's own definition unchanged (ABI v5:
 * runtime/virtual/heap.c over Capstone-compiled musl mallocng, in the program itself).
 * Because the program is linked statically, --wrap also catches the calls musl's capability
 * libc makes (strdup, getline, ...); an allocator function reached from inside another
 * wrapped call is not counted again (busy).
 *
 * Output, on stderr:
 *   MQ op= live_req= live_asked= fresh_addr= reused_addr=
 *   MQ-HIST b:count ...   MQ-SIZES b:count ...
 *   MQ-DONE objects= ops= peak_live= peak_live_objects= peak_live_pow2_256=
 *           heap_allocations= heap_frees= heap_peak_objects=
 * op counts allocations; a line every MQ_SAMPLE allocations (default 4096), and every
 * MQ_HIST_EVERY samples (default 256) also the MQ-HIST and MQ-SIZES lines so far, so that
 * a run stopped at a time limit still reports them.
 *   live_req    usable bytes of the live objects (malloc_usable_size; ABI v5 reports the
 *               requested size, so live_req = live_asked)
 *   live_asked  the bytes the program asked for, counted as the usable size at free
 * The live counts need no table: an object's size at free is malloc_usable_size, and the
 * live-object count is allocations minus frees. A free of an object allocated before the
 * constructor ran is counted too (the count is clamped at zero), an error of a few objects.
 *   peak_live_pow2_256  peak of the live bytes if each object were rounded up to a power of
 *               two of at least 256 bytes (as on CheriBSD, for the Sublet-fit comparison)
 *   MQ-HIST     reuse distance: allocations between the free of an address and its
 *               next allocation, log2 buckets (b = 64 - clz(distance), 0 = none between).
 *               With MQ_ADDR_SAMPLE=k (default 1) only addresses whose hash falls in 1 of
 *               k buckets are followed, each with all its reuses (as the CheriBSD tracer's
 *               MQ_TRACK=3); fresh_addr and reused_addr then cover the followed addresses.
 *   MQ-SIZES    log2 histogram of the asked sizes, same buckets
 * heap_* are the trusted heap's own counters (__capstone_sublet_heap_stats, service 13):
 * objects should equal heap_allocations less the allocations made before the constructor
 * ran, which checks that the wrappers see every allocation once.
 * The tables are mmap'd before counting starts but are resident: footprint runs use the
 * untraced build. Single-threaded programs only (no locking).
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);
int __real_posix_memalign(void **, size_t, size_t);
void *__real_aligned_alloc(size_t, size_t);
size_t malloc_usable_size(void *);
void __capstone_sublet_heap_stats(unsigned long out[9]);

#define TBITS 22
#define TSIZE (1ul << TBITS)

static int busy, ready;
static uint64_t op, every = 4096, hist_every = 256, samples, objects, live, peak_live, asked;
static uint64_t live_objs, peak_live_objs, live_pow2, peak_live_pow2;
static uint64_t *tkey, *tfree, seen, fresh, reused, hist[65], sizes[65], addr_keep = 1;
static const char *pfx = "MQ";

static uint64_t pow2_256(uint64_t n)
{
    uint64_t b = 256;
    while (b < n) b <<= 1;
    return b;
}
static uint64_t addr(void *p) { return __builtin_capstone_cap_get_cursor(p); }
static uint64_t slot(uint64_t a) { return (a >> 4) * 0x9e3779b97f4a7c15ull >> (64 - TBITS); }
static int followed(uint64_t a) { return ((a >> 4) * 0xd6e8feb86659fd93ull >> 32) % addr_keep == 0; }

static void histograms(void)
{
    char line[2048];
    int n = snprintf(line, sizeof line, "%s-HIST", pfx);
    for (int b = 0; b < 65 && n > 0 && (size_t)n < sizeof line - 32; b++)
        if (hist[b]) n += snprintf(line + n, sizeof line - n, " %d:%llu", b, (unsigned long long)hist[b]);
    n += snprintf(line + n, sizeof line - n, "\n%s-SIZES", pfx);
    for (int b = 0; b < 65 && n > 0 && (size_t)n < sizeof line - 200; b++)
        if (sizes[b]) n += snprintf(line + n, sizeof line - n, " %d:%llu", b, (unsigned long long)sizes[b]);
    n += snprintf(line + n, sizeof line - n, "\n");
    if (n > 0) write(2, line, (size_t)n);
}

static void sample(void)
{
    char line[256];
    int n = snprintf(line, sizeof line,
        "%s op=%llu live_req=%llu live_asked=%llu fresh_addr=%llu reused_addr=%llu\n", pfx,
        (unsigned long long)op, (unsigned long long)live, (unsigned long long)asked,
        (unsigned long long)fresh, (unsigned long long)reused);
    if (n > 0) write(2, line, (size_t)n);
    if (++samples % hist_every == 0) histograms();
}

static void got(void *p, uint64_t n)
{
    if (!p || busy || !ready) return;
    busy++;
    uint64_t a = addr(p), sz = malloc_usable_size(p);
    op++; objects++;
    live += sz; if (live > peak_live) peak_live = live;
    asked += n;
    sizes[n ? 64 - __builtin_clzll(n) : 0]++;
    if (++live_objs > peak_live_objs) peak_live_objs = live_objs;
    live_pow2 += pow2_256(sz); if (live_pow2 > peak_live_pow2) peak_live_pow2 = live_pow2;
    if (followed(a))
        for (uint64_t h = slot(a);; h = (h + 1) & (TSIZE - 1)) {
            if (tkey[h] == a) {
                if (tfree[h]) {
                    uint64_t r = op - tfree[h];
                    hist[r ? 64 - __builtin_clzll(r) : 0]++;
                    reused++; tfree[h] = 0;
                }
                break;
            }
            if (!tkey[h]) {
                if (++seen > TSIZE / 2) { write(2, "MQ-ERROR table full\n", 20); _exit(3); }
                tkey[h] = a; tfree[h] = 0; fresh++;
                break;
            }
        }
    if (op % every == 0) sample();
    busy--;
}

static void gone(void *p)
{
    if (!p || busy || !ready) return;
    busy++;
    uint64_t a = addr(p), sz = malloc_usable_size(p), r = pow2_256(sz);
    live = live >= sz ? live - sz : 0;
    asked = asked >= sz ? asked - sz : 0;
    live_pow2 = live_pow2 >= r ? live_pow2 - r : 0;
    if (live_objs) live_objs--;
    if (followed(a))
        for (uint64_t h = slot(a);; h = (h + 1) & (TSIZE - 1)) {
            if (tkey[h] == a) { tfree[h] = op ? op : 1; break; }
            if (!tkey[h]) break; /* allocated before tracking began */
        }
    busy--;
}

static uint64_t *table(void)
{
    void *p = mmap(0, TSIZE * 8, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { write(2, "MQ-ERROR mmap\n", 14); _exit(3); }
    return p;
}

__attribute__((constructor)) static void mq_init(void)
{
    const char *s = getenv("MQ_SAMPLE"), *x = getenv("MQ_PREFIX"), *k = getenv("MQ_ADDR_SAMPLE");
    const char *h = getenv("MQ_HIST_EVERY");
    if (k && atoll(k) > 0) addr_keep = (uint64_t)atoll(k);
    if (h && atoll(h) > 0) hist_every = (uint64_t)atoll(h);
    if (x && *x && strlen(x) < 8) pfx = x;
    if (s && atoll(s) > 0) every = (uint64_t)atoll(s);
    tkey = table(); tfree = table();
    ready = 1;
    sample();
}

__attribute__((destructor)) static void mq_fini(void)
{
    if (!ready) return;
    busy++;
    sample();
    histograms();
    char line[512];
    int n = 0;
    unsigned long hs[9];
    __capstone_sublet_heap_stats(hs);
    n += snprintf(line + n, sizeof line - n, "%s-DONE objects=%llu ops=%llu peak_live=%llu "
                  "peak_live_objects=%llu peak_live_pow2_256=%llu "
                  "heap_allocations=%lu heap_frees=%lu heap_peak_objects=%lu\n", pfx,
                  (unsigned long long)objects, (unsigned long long)op, (unsigned long long)peak_live,
                  (unsigned long long)peak_live_objs, (unsigned long long)peak_live_pow2,
                  hs[0], hs[1], hs[3]);
    n += snprintf(line + n, sizeof line - n, "%s-ADDR addr_sample=%llu\n", pfx,
                  (unsigned long long)addr_keep);
    write(2, line, (size_t)n);
    busy--;
}

/* busy is held across each forwarded call, so an allocation the heap makes through another
 * wrapped name is not counted twice. */
#define FORWARD(call) (busy++, _r = (call), busy--, _r)
void *__wrap_malloc(size_t n) { void *_r; void *p = FORWARD(__real_malloc(n)); got(p, n); return p; }
void *__wrap_calloc(size_t a, size_t b) { void *_r; void *p = FORWARD(__real_calloc(a, b)); got(p, a * b); return p; }
void __wrap_free(void *p) { gone(p); busy++; __real_free(p); busy--; }
void *__wrap_realloc(void *old, size_t n)
{
    void *_r;
    if (old) gone(old);   /* realloc(p, 0) frees inside the heap, unwrapped */
    void *p = FORWARD(__real_realloc(old, n));
    if (p) got(p, n);
    else if (old && n) got(old, n); /* failed: the old object is still live */
    return p;
}
int __wrap_posix_memalign(void **out, size_t al, size_t n)
{
    int _r; int r = FORWARD(__real_posix_memalign(out, al, n));
    if (!r) got(*out, n);
    return r;
}
void *__wrap_aligned_alloc(size_t al, size_t n) { void *_r; void *p = FORWARD(__real_aligned_alloc(al, n)); got(p, n); return p; }
