/* ffcensus: slotcensus for Firefox, as a replace-malloc library (Nightly builds; on Linux firefox-bin takes the weak
 * symbol replace_init from a preloaded library, which is all this one exports).
 *
 * Firefox allocates through its own mozjemalloc, which an LD_PRELOAD malloc tracer cannot see; replace-malloc hands
 * this library mozjemalloc's function table at start-up, and it puts its counting functions in front of the originals:
 * malloc, calloc, realloc, free, memalign, posix_memalign, aligned_alloc, valloc and the moz_arena_* family. The
 * counting is slotcensus.c's (live, peak by compare-and-swap, allocs, frees, realloc moves and in-place resizes,
 * failures, live and peak usable bytes, log2 histogram of requested sizes; a fork child starts at the live set it
 * inherits, with its own counts).
 *
 * Firefox ends child processes with _exit or a kill, so nothing runs at exit. The counters therefore live in a file
 * mapped MAP_SHARED, $FFCENSUS_DIR/census.<pid>, which the kernel keeps whatever way the process ends; ffcensus-dump.py
 * turns the files into slotcensus's JSON lines. The command line and thread name are stored at start-up and right
 * after a fork, never later: Firefox's seccomp sandbox kills a content process that opens a file (SIGSYS, measured
 * when this library re-read /proc/self/cmdline during the run).
 * JavaScript objects live in SpiderMonkey's garbage-collected heap, not in malloc: they are hidden from this census,
 * as memcached's items are. Built as C against the replace-malloc headers of the exact Nightly revision. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

#include "replace_malloc.h"

#define FFCENSUS_MAGIC 0x53554e4543534646ull   /* "FFCENSUS" */

struct census {
    uint64_t magic;
    int32_t pid, ppid, forked, pad;
    int64_t live, peak, allocs, frees, moved, inplace, failed, live_bytes, peak_bytes;
    int64_t size_log2[64];
    char comm[32];
    char cmdline[1024];
};

static malloc_table_t orig;
static struct census *c;
static char dir[512];

static void name_process(void)
{
    char buf[1024];
    int fd = open("/proc/self/cmdline", O_RDONLY);
    if (fd >= 0) {
        ssize_t n = read(fd, buf, sizeof buf - 1);
        close(fd);
        n = n > 0 ? n : 0;
        while (n > 0 && buf[n - 1] == 0) --n;
        for (ssize_t i = 0; i < n; ++i)
            buf[i] = buf[i] == 0 ? ' ' : buf[i] == '"' || buf[i] == '\\' || (unsigned char)buf[i] < 0x20 ? '_' : buf[i];
        buf[n] = 0;
        memcpy(c->cmdline, buf, (size_t)n + 1);
    }
    char comm[32] = {0};
    prctl(PR_GET_NAME, comm, 0, 0, 0);
    for (char *p = comm; *p; ++p)
        if (*p == '"' || *p == '\\' || (unsigned char)*p < 0x20) *p = '_';
    memcpy(c->comm, comm, sizeof comm);
}

/* a fresh per-process file; the counters start from what the caller copies in */
static struct census *open_census(const struct census *from)
{
    char path[600];
    snprintf(path, sizeof path, "%s/census.%d", dir, (int)getpid());
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || ftruncate(fd, sizeof(struct census)) != 0)
        return NULL;
    struct census *m = mmap(NULL, sizeof(struct census), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED)
        return NULL;
    if (from)
        memcpy(m, from, sizeof *m);
    m->pid = (int32_t)getpid();
    m->ppid = (int32_t)getppid();
    m->magic = FFCENSUS_MAGIC;
    return m;
}

static void raise_max(int64_t *max, int64_t v)
{
    int64_t old = __atomic_load_n(max, __ATOMIC_RELAXED);
    while (v > old && !__atomic_compare_exchange_n(max, &old, v, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

static void note_size(size_t n)
{
    unsigned b = n > 1 ? 63 - (unsigned)__builtin_clzll((unsigned long long)n) : 0;
    __atomic_add_fetch(&c->size_log2[b], 1, __ATOMIC_RELAXED);
}

static void note_alloc(void *p, size_t requested)
{
    if (!p) {
        __atomic_add_fetch(&c->failed, 1, __ATOMIC_RELAXED);
        return;
    }
    __atomic_add_fetch(&c->allocs, 1, __ATOMIC_RELAXED);
    note_size(requested);
    raise_max(&c->peak, __atomic_add_fetch(&c->live, 1, __ATOMIC_RELAXED));
    int64_t usable = (int64_t)orig.malloc_usable_size(p);
    raise_max(&c->peak_bytes, __atomic_add_fetch(&c->live_bytes, usable, __ATOMIC_RELAXED));
}

static void note_free(void *p)
{
    if (!p)
        return;
    __atomic_add_fetch(&c->frees, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&c->live, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&c->live_bytes, (int64_t)orig.malloc_usable_size(p), __ATOMIC_RELAXED);
}

/* realloc and moz_arena_realloc share this: o is the old object (non-null), p the result, was its old usable size */
static void note_realloc(void *o, void *p, size_t n, size_t was)
{
    if (!p) {
        if (n == 0) {
            __atomic_add_fetch(&c->frees, 1, __ATOMIC_RELAXED);
            __atomic_sub_fetch(&c->live, 1, __ATOMIC_RELAXED);
            __atomic_sub_fetch(&c->live_bytes, (int64_t)was, __ATOMIC_RELAXED);
        } else {
            __atomic_add_fetch(&c->failed, 1, __ATOMIC_RELAXED);
        }
        return;
    }
    note_size(n);
    if (p == o) {
        __atomic_add_fetch(&c->inplace, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_add_fetch(&c->moved, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&c->allocs, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&c->frees, 1, __ATOMIC_RELAXED);
    }
    raise_max(&c->peak_bytes, __atomic_add_fetch(&c->live_bytes,
                                                 (int64_t)orig.malloc_usable_size(p) - (int64_t)was, __ATOMIC_RELAXED));
}

static void *census_malloc(size_t n) { void *p = orig.malloc(n); note_alloc(p, n); return p; }
static void *census_calloc(size_t a, size_t b) { void *p = orig.calloc(a, b); note_alloc(p, a * b); return p; }
static void census_free(void *p) { note_free(p); orig.free(p); }
static void *census_memalign(size_t al, size_t n) { void *p = orig.memalign(al, n); note_alloc(p, n); return p; }
static void *census_aligned_alloc(size_t al, size_t n)
{
    void *p = orig.aligned_alloc(al, n);
    note_alloc(p, n);
    return p;
}
static void *census_valloc(size_t n) { void *p = orig.valloc(n); note_alloc(p, n); return p; }

static int census_posix_memalign(void **out, size_t al, size_t n)
{
    int rc = orig.posix_memalign(out, al, n);
    note_alloc(rc == 0 ? *out : NULL, n);
    return rc;
}

static void *census_realloc(void *o, size_t n)
{
    if (!o) { void *p = orig.realloc(NULL, n); note_alloc(p, n); return p; }
    size_t was = orig.malloc_usable_size(o);
    void *p = orig.realloc(o, n);
    note_realloc(o, p, n, was);
    return p;
}

static void *census_arena_malloc(arena_id_t a, size_t n)
{
    void *p = orig.moz_arena_malloc(a, n);
    note_alloc(p, n);
    return p;
}
static void census_arena_free(arena_id_t a, void *p) { note_free(p); orig.moz_arena_free(a, p); }

static void *census_arena_calloc(arena_id_t a, size_t k, size_t n)
{
    void *p = orig.moz_arena_calloc(a, k, n);
    note_alloc(p, k * n);
    return p;
}

static void *census_arena_memalign(arena_id_t a, size_t al, size_t n)
{
    void *p = orig.moz_arena_memalign(a, al, n);
    note_alloc(p, n);
    return p;
}

static void *census_arena_realloc(arena_id_t a, void *o, size_t n)
{
    if (!o) { void *p = orig.moz_arena_realloc(a, NULL, n); note_alloc(p, n); return p; }
    size_t was = orig.malloc_usable_size(o);
    void *p = orig.moz_arena_realloc(a, o, n);
    note_realloc(o, p, n, was);
    return p;
}

/* Arena creation and disposal pass straight through, but must be replaced too: with the moz_arena_* allocation
 * functions replaced and these left alone, Firefox crashed in its first moz_arena_malloc (measured; with both
 * replaced it runs). */
static arena_id_t census_create_arena(arena_params_t *params) { return orig.moz_create_arena_with_params(params); }
static void census_dispose_arena(arena_id_t a) { orig.moz_dispose_arena(a); }

/* a fork child starts at the parent's live set, with its own file and counts (as slotcensus.c's child) */
static void child_after_fork(void)
{
    struct census snap = *c;
    snap.forked = 1;
    snap.allocs = snap.frees = snap.moved = snap.inplace = snap.failed = 0;
    memset(snap.size_log2, 0, sizeof snap.size_log2);
    snap.peak = snap.live;
    snap.peak_bytes = snap.live_bytes;
    struct census *m = open_census(&snap);
    if (m)
        c = m;   /* the parent's mapping stays mapped: the old pointer must not be written again */
    name_process();
}

void replace_init(malloc_table_t *table, struct ReplaceMallocBridge **bridge)
{
    (void)bridge;
    const char *d = getenv("FFCENSUS_DIR");
    if (!d || strlen(d) >= sizeof dir)
        return;   /* no directory: replace nothing, so the run cannot pass unmeasured */
    strcpy(dir, d);
    c = open_census(NULL);
    if (!c)
        return;
    name_process();
    orig = *table;
    table->malloc = census_malloc;
    table->calloc = census_calloc;
    table->realloc = census_realloc;
    table->free = census_free;
    table->memalign = census_memalign;
    table->posix_memalign = census_posix_memalign;
    table->aligned_alloc = census_aligned_alloc;
    table->valloc = census_valloc;
    table->moz_create_arena_with_params = census_create_arena;
    table->moz_dispose_arena = census_dispose_arena;
    table->moz_arena_malloc = census_arena_malloc;
    table->moz_arena_calloc = census_arena_calloc;
    table->moz_arena_realloc = census_arena_realloc;
    table->moz_arena_free = census_arena_free;
    table->moz_arena_memalign = census_arena_memalign;
    pthread_atfork(NULL, NULL, child_after_fork);
}
