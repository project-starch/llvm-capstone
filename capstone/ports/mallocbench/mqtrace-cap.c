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
 * Per sample, after the fields above (all cumulative unless named live):
 *   ids_avail   lifetime IDs the processor reports available (urevavail)
 *   heap_alloc heap_free heap_split heap_mrev heap_delin heap_revoke heap_init
 *               the heap's own operation counters (__capstone_sublet_heap_stats)
 *   MQ-LIFE     object lifetime: allocations between the allocation of an object and its
 *               free, log2 buckets as MQ-HIST, over the followed addresses; printed with
 *               MQ-HIST. MQ-DONE adds unfreed= (followed objects still live at exit).
 *   MQ-STRIDE   distance in bytes between consecutive allocations, log2 buckets (as the
 *               CheriBSD tracer's stride histogram)
 * Built with -DMQ_NATIVE (native/build-native.sh) the same file traces native musl: no
 * live bytes (see usable()), ids_avail and heap_* read 0.
 * Runtime allocations: the virtual pthread bridge allocates its own per-thread records
 * (runtime/virtual/pthread.c: __clone, and the child's attach and detach), and --wrap sees
 * those calls too, while CheriBSD's tracer cannot see libthr's and native musl makes none.
 * Allocations made inside __clone, __capstone_delegate_thread_attach and
 * __capstone_signals_thread_detach are therefore not counted, and their addresses are kept
 * until the matching free, which is not counted either. They remain in RSS and pinned pages,
 * as the platform's cost. MQ-DONE reports how many were set aside (runtime_allocs=).
 * Threads: every counter and table is updated under one spin lock, and the guard against
 * counting the heap's own nested calls is per thread.
 * heap_* are the trusted heap's own counters (__capstone_sublet_heap_stats, service 13):
 * objects should equal heap_allocations less the allocations made before the constructor
 * ran, which checks that the wrappers see every allocation once.
 * The tables are mmap'd before counting starts but are resident: footprint runs use the
 * untraced build. 
 */
#include <stdarg.h>
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

static __thread int busy;
static int ready, lock_word;
static uint64_t op, every = 4096, hist_every = 256, samples, objects, live, peak_live, asked;
static uint64_t live_objs, peak_live_objs, live_pow2, peak_live_pow2;
static uint64_t *tkey, *tfree, *talloc, seen, fresh, reused, hist[65], sizes[65], addr_keep = 1;
static uint64_t life[65], stride[65], last_addr, unfreed;
static __thread int in_runtime;
#define IGN 1024
static uint64_t ign[IGN], ign_live, runtime_allocs, runtime_frees;
static void ign_add(uint64_t a)
{
    for (int i = 0; i < IGN; i++) if (!ign[i]) { ign[i] = a; ign_live++; runtime_allocs++; return; }
    write(2, "MQ-ERROR runtime table full\n", 28); _exit(3);
}
/* Called on every free: the table is searched only while it holds an address, which for a
 * single-threaded program is never (a search per free made traced runs 8x slower). */
static int ign_take(uint64_t a)
{
    if (!ign_live) return 0;
    for (int i = 0; i < IGN; i++) if (ign[i] == a) { ign[i] = 0; ign_live--; runtime_frees++; return 1; }
    return 0;
}
static void lock(void) { while (__atomic_test_and_set(&lock_word, __ATOMIC_ACQUIRE)) ; }
static void unlock(void) { __atomic_clear(&lock_word, __ATOMIC_RELEASE); }
static uint64_t bucket(uint64_t r) { return r ? 64 - __builtin_clzll(r) : 0; }
static unsigned long ids_avail(void)
{
#ifdef MQ_NATIVE
    return 0;
#else
    unsigned long v;
    __asm__ volatile("csrr %0, 0xcc0" : "=r"(v));
    return v;
#endif
}
/* Capstone's malloc_usable_size is the requested size (ABI v5); native mallocng reports its
 * slot size, which would not compare, so the native build tracks no live bytes. */
static uint64_t usable(void *p)
{
#ifdef MQ_NATIVE
    (void)p;
    return 0;
#else
    return malloc_usable_size(p);
#endif
}
static const char *pfx = "MQ";

static uint64_t pow2_256(uint64_t n)
{
    uint64_t b = 256;
    while (b < n) b <<= 1;
    return b;
}
#ifdef MQ_NATIVE
static uint64_t addr(void *p) { return (uint64_t)(uintptr_t)p; }
void __capstone_sublet_heap_stats(unsigned long out[9]) { memset(out, 0, 9 * sizeof *out); }
#else
static uint64_t addr(void *p) { return __builtin_capstone_cap_get_cursor(p); }
#endif
static uint64_t slot(uint64_t a) { return (a >> 4) * 0x9e3779b97f4a7c15ull >> (64 - TBITS); }
static int followed(uint64_t a) { return ((a >> 4) * 0xd6e8feb86659fd93ull >> 32) % addr_keep == 0; }

static void histograms(void)
{
    char line[4096];
    int n = snprintf(line, sizeof line, "%s-HIST", pfx);
    for (int b = 0; b < 65 && n > 0 && (size_t)n < sizeof line - 32; b++)
        if (hist[b]) n += snprintf(line + n, sizeof line - n, " %d:%llu", b, (unsigned long long)hist[b]);
    n += snprintf(line + n, sizeof line - n, "\n%s-SIZES", pfx);
    for (int b = 0; b < 65 && n > 0 && (size_t)n < sizeof line - 200; b++)
        if (sizes[b]) n += snprintf(line + n, sizeof line - n, " %d:%llu", b, (unsigned long long)sizes[b]);
    n += snprintf(line + n, sizeof line - n, "\n%s-LIFE", pfx);
    for (int b = 0; b < 65 && n > 0 && (size_t)n < sizeof line - 200; b++)
        if (life[b]) n += snprintf(line + n, sizeof line - n, " %d:%llu", b, (unsigned long long)life[b]);
    n += snprintf(line + n, sizeof line - n, "\n%s-STRIDE", pfx);
    for (int b = 0; b < 65 && n > 0 && (size_t)n < sizeof line - 200; b++)
        if (stride[b]) n += snprintf(line + n, sizeof line - n, " %d:%llu", b, (unsigned long long)stride[b]);
    n += snprintf(line + n, sizeof line - n, "\n");
    if (n > 0) write(2, line, (size_t)n);
}

static void sample(void)
{
    char line[512];
    unsigned long hs[9];
    __capstone_sublet_heap_stats(hs);
    int n = snprintf(line, sizeof line,
        "%s op=%llu live_req=%llu live_asked=%llu fresh_addr=%llu reused_addr=%llu "
        "live_objects=%llu ids_avail=%lu heap_alloc=%lu heap_free=%lu heap_split=%lu "
        "heap_mrev=%lu heap_delin=%lu heap_revoke=%lu heap_init=%lu\n", pfx,
        (unsigned long long)op, (unsigned long long)live, (unsigned long long)asked,
        (unsigned long long)fresh, (unsigned long long)reused, (unsigned long long)live_objs,
        ids_avail(), hs[0], hs[1], hs[4], hs[5], hs[6], hs[7], hs[8]);
    if (n > 0) write(2, line, (size_t)n);
    if (++samples % hist_every == 0) histograms();
}

static void got(void *p, uint64_t n)
{
    if (!p || !ready || busy) return;
    busy++;
    uint64_t a = addr(p), sz = usable(p);
    lock();
    if (in_runtime) { ign_add(a); unlock(); busy--; return; }
    op++; objects++;
    if (last_addr) stride[bucket(a > last_addr ? a - last_addr : last_addr - a)]++;
    last_addr = a;
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
                if (!talloc[h]) unfreed++;
                talloc[h] = op;
                break;
            }
            if (!tkey[h]) {
                if (++seen > TSIZE / 2) { write(2, "MQ-ERROR table full\n", 20); _exit(3); }
                tkey[h] = a; tfree[h] = 0; talloc[h] = op; fresh++; unfreed++;
                break;
            }
        }
    if (op % every == 0) sample();
    unlock();
    busy--;
}

static void gone(void *p)
{
    if (!p || !ready || busy) return;
    busy++;
    uint64_t a = addr(p), sz = usable(p), r = pow2_256(sz);
    lock();
    if (ign_take(a)) { unlock(); busy--; return; }
    live = live >= sz ? live - sz : 0;
    asked = asked >= sz ? asked - sz : 0;
    live_pow2 = live_pow2 >= r ? live_pow2 - r : 0;
    if (live_objs) live_objs--;
    if (followed(a))
        for (uint64_t h = slot(a);; h = (h + 1) & (TSIZE - 1)) {
            if (tkey[h] == a) {
                if (talloc[h]) { life[bucket(op - talloc[h])]++; talloc[h] = 0; unfreed--; }
                tfree[h] = op ? op : 1;
                break;
            }
            if (!tkey[h]) break; /* allocated before tracking began */
        }
    unlock();
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
    tkey = table(); tfree = table(); talloc = table();
    ready = 1;
    sample();
}

__attribute__((destructor)) static void mq_fini(void)
{
    if (!ready) return;
    busy++;
    lock();
    sample();
    histograms();
    char line[512];
    int n = 0;
    unsigned long hs[9];
    __capstone_sublet_heap_stats(hs);
    n += snprintf(line + n, sizeof line - n, "%s-DONE objects=%llu ops=%llu peak_live=%llu "
                  "peak_live_objects=%llu peak_live_pow2_256=%llu "
                  "heap_allocations=%lu heap_frees=%lu heap_peak_objects=%lu unfreed=%llu "
                  "runtime_allocs=%llu runtime_frees=%llu\n", pfx,
                  (unsigned long long)objects, (unsigned long long)op, (unsigned long long)peak_live,
                  (unsigned long long)peak_live_objs, (unsigned long long)peak_live_pow2,
                  hs[0], hs[1], hs[3], (unsigned long long)unfreed,
                  (unsigned long long)runtime_allocs, (unsigned long long)runtime_frees);
    n += snprintf(line + n, sizeof line - n, "%s-ADDR addr_sample=%llu\n", pfx,
                  (unsigned long long)addr_keep);
    write(2, line, (size_t)n);
    unlock();
    busy--;
}

/* busy is held across each forwarded call, so an allocation the heap makes through another
 * wrapped name is not counted twice. Until the constructor has run, the wrappers only forward:
 * the runtime allocates the first thread's TLS block with calloc before tp exists, and busy is
 * thread-local. */
#define FORWARD(call) (busy++, _r = (call), busy--, _r)
void *__wrap_malloc(size_t n)
{
    void *_r;
    if (!ready) return __real_malloc(n);
    void *p = FORWARD(__real_malloc(n)); got(p, n); return p;
}
void *__wrap_calloc(size_t a, size_t b)
{
    void *_r;
    if (!ready) return __real_calloc(a, b);
    void *p = FORWARD(__real_calloc(a, b)); got(p, a * b); return p;
}
void __wrap_free(void *p)
{
    if (!ready) { __real_free(p); return; }
    gone(p); busy++; __real_free(p); busy--;
}
void *__wrap_realloc(void *old, size_t n)
{
    void *_r;
    if (!ready) return __real_realloc(old, n);
    if (old) gone(old);   /* realloc(p, 0) frees inside the heap, unwrapped */
    void *p = FORWARD(__real_realloc(old, n));
    if (p) got(p, n);
    else if (old && n) got(old, n); /* failed: the old object is still live */
    return p;
}
int __wrap_posix_memalign(void **out, size_t al, size_t n)
{
    int _r;
    if (!ready) return __real_posix_memalign(out, al, n);
    int r = FORWARD(__real_posix_memalign(out, al, n));
    if (!r) got(*out, n);
    return r;
}
#ifndef MQ_NATIVE
int __real___clone(int (*)(void *), void *, int, void *, ...);
void __real___capstone_delegate_thread_attach(void *, void *, void *, void *);
void __real___capstone_signals_thread_detach(void);
/* musl's pthread_create passes exactly three variadic arguments to __clone (parent tid
 * word, TLS, clear word); the wrapper is variadic too, as the caller's prototype is. */
int __wrap___clone(int (*entry)(void *), void *stack, int flags, void *arg, ...)
{
    va_list ap;
    va_start(ap, arg);
    int *parent = va_arg(ap, int *);
    void *tls = va_arg(ap, void *);
    int *clear = va_arg(ap, int *);
    va_end(ap);
    in_runtime++;
    int r = __real___clone(entry, stack, flags, arg, parent, tls, clear);
    in_runtime--;
    return r;
}
void __wrap___capstone_delegate_thread_attach(void *t, void *m, void *e, void *s)
{
    in_runtime++;
    __real___capstone_delegate_thread_attach(t, m, e, s);
    in_runtime--;
}
void __wrap___capstone_signals_thread_detach(void)
{
    in_runtime++;
    __real___capstone_signals_thread_detach();
    in_runtime--;
}
#endif
void *__wrap_aligned_alloc(size_t al, size_t n)
{
    void *_r;
    if (!ready) return __real_aligned_alloc(al, n);
    void *p = FORWARD(__real_aligned_alloc(al, n)); got(p, n); return p;
}
