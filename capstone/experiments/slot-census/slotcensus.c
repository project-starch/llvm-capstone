/* slotcensus: an LD_PRELOAD census of simultaneously live heap objects, for sizing lifetime-id slots.
 *
 * Exact at every event: every successful allocation and free updates one live counter, and the peak is the running
 * maximum of that counter (compare-and-swap), never a sample. Counted entry points: malloc, calloc, realloc,
 * reallocarray, free, posix_memalign, aligned_alloc, memalign, valloc, pvalloc (C++ operator new/delete reach malloc
 * and free through libstdc++). realloc: NULL old pointer = an allocation; size 0 that frees = a free; a result at the
 * old address = in place (no change to the live count); a result elsewhere = a move (one free, one allocation, the
 * live count unchanged; counted as moved). malloc(0) returns a unique pointer, so it is an object.
 *
 * Per process, at exit, one JSON line is appended to $SLOTCENSUS_OUT:
 *   pid, ppid, cmdline (the first 1023 bytes of /proc/self/cmdline, arguments joined by spaces), forked (1 if this
 *   process began as a fork child, which inherits the parent's live objects),
 *   peak_live, end_live, allocs, frees, realloc_moved, realloc_inplace, failed, peak_live_bytes (usable bytes, see
 *   below), size_log2 (requested sizes of every allocation: count per bucket floor(log2(size)); bucket 0 holds 0
 *   and 1).
 * Bytes are malloc_usable_size, so they include the allocator's rounding: peak_live_bytes is an upper bound on the
 * requested bytes live at the byte peak, and the byte peak need not coincide with the object peak.
 * Without SLOTCENSUS_OUT nothing is written. The path is read once at start-up (and kept across fork), because some
 * programs clear the environment of the processes they fork (nginx's workers keep only the variables its env
 * directive names): a path read at exit would be gone, and those processes would silently not report. Allocations
 * made by dlsym before the real functions are resolved come from a static buffer and are not counted. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void *(*r_malloc)(size_t);
static void *(*r_calloc)(size_t, size_t);
static void *(*r_realloc)(void *, size_t);
static void (*r_free)(void *);
static int (*r_posix_memalign)(void **, size_t, size_t);
static void *(*r_aligned_alloc)(size_t, size_t);
static void *(*r_memalign)(size_t, size_t);

static long long live, peak, allocs, frees, moved, inplace, failed;
static long long live_bytes, peak_bytes;
static long long size_log2[64];
static int forked;

static char out_path[4096];   /* $SLOTCENSUS_OUT, read at start-up: nginx and others clear a child's environment */

static char boot[1 << 16];
static size_t boot_used;
static int resolving;

static int in_boot(const void *p) { return (const char *)p >= boot && (const char *)p < boot + sizeof boot; }

static void resolve(void)
{
    if (r_malloc || resolving)
        return;
    resolving = 1;
    r_malloc = dlsym(RTLD_NEXT, "malloc");
    r_calloc = dlsym(RTLD_NEXT, "calloc");
    r_realloc = dlsym(RTLD_NEXT, "realloc");
    r_free = dlsym(RTLD_NEXT, "free");
    r_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
    r_aligned_alloc = dlsym(RTLD_NEXT, "aligned_alloc");
    r_memalign = dlsym(RTLD_NEXT, "memalign");
    resolving = 0;
}

static void raise_max(long long *max, long long v)
{
    long long old = __atomic_load_n(max, __ATOMIC_RELAXED);
    while (v > old && !__atomic_compare_exchange_n(max, &old, v, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

static void note_size(size_t n)
{
    unsigned b = n > 1 ? 63 - (unsigned)__builtin_clzll((unsigned long long)n) : 0;
    __atomic_add_fetch(&size_log2[b], 1, __ATOMIC_RELAXED);
}

static void note_alloc(void *p, size_t requested)
{
    if (!p) {
        __atomic_add_fetch(&failed, 1, __ATOMIC_RELAXED);
        return;
    }
    __atomic_add_fetch(&allocs, 1, __ATOMIC_RELAXED);
    note_size(requested);
    raise_max(&peak, __atomic_add_fetch(&live, 1, __ATOMIC_RELAXED));
    raise_max(&peak_bytes, __atomic_add_fetch(&live_bytes, (long long)malloc_usable_size(p), __ATOMIC_RELAXED));
}

static void note_free(void *p)
{
    if (!p || in_boot(p))
        return;
    __atomic_add_fetch(&frees, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&live, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&live_bytes, (long long)malloc_usable_size(p), __ATOMIC_RELAXED);
}

void *malloc(size_t n) { resolve(); void *p = r_malloc(n); note_alloc(p, n); return p; }

void *calloc(size_t a, size_t b)
{
    if (!r_calloc) {
        resolve();
        if (!r_calloc) {   /* inside dlsym: serve from the boot buffer, uncounted */
            size_t n = (a * b + 15) & ~(size_t)15;
            if (boot_used + n > sizeof boot)
                return NULL;
            void *p = boot + boot_used;
            boot_used += n;
            return p;      /* static storage is zero */
        }
    }
    void *p = r_calloc(a, b);
    note_alloc(p, a * b);
    return p;
}

void *realloc(void *o, size_t n)
{
    resolve();
    if (!o) { void *p = r_realloc(NULL, n); note_alloc(p, n); return p; }
    if (in_boot(o)) {
        void *p = r_malloc(n);
        if (p) memcpy(p, o, n);
        note_alloc(p, n);
        return p;
    }
    size_t was = malloc_usable_size(o);
    void *p = r_realloc(o, n);
    if (!p) {
        if (n == 0) {   /* glibc's realloc(p, 0) frees p and returns NULL */
            __atomic_add_fetch(&frees, 1, __ATOMIC_RELAXED);
            __atomic_sub_fetch(&live, 1, __ATOMIC_RELAXED);
            __atomic_sub_fetch(&live_bytes, (long long)was, __ATOMIC_RELAXED);
        } else {        /* failed: the old object is untouched */
            __atomic_add_fetch(&failed, 1, __ATOMIC_RELAXED);
        }
        return p;
    }
    note_size(n);
    if (p == o) {
        __atomic_add_fetch(&inplace, 1, __ATOMIC_RELAXED);
    } else {
        /* a move is one free and one allocation; the live count does not change */
        __atomic_add_fetch(&moved, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&allocs, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&frees, 1, __ATOMIC_RELAXED);
    }
    raise_max(&peak_bytes, __atomic_add_fetch(&live_bytes, (long long)malloc_usable_size(p) - (long long)was,
                                              __ATOMIC_RELAXED));
    return p;
}

void *reallocarray(void *o, size_t a, size_t b)
{
    if (b && a > SIZE_MAX / b) return NULL;
    return realloc(o, a * b);
}

void free(void *p)
{
    if (in_boot(p))
        return;
    resolve();
    note_free(p);
    r_free(p);
}

int posix_memalign(void **out, size_t align, size_t n)
{
    resolve();
    int rc = r_posix_memalign(out, align, n);
    note_alloc(rc == 0 ? *out : NULL, n);
    return rc;
}

void *aligned_alloc(size_t align, size_t n)
{
    resolve();
    void *p = r_aligned_alloc(align, n);
    note_alloc(p, n);
    return p;
}
void *memalign(size_t align, size_t n) { resolve(); void *p = r_memalign(align, n); note_alloc(p, n); return p; }
void *valloc(size_t n) { return memalign((size_t)sysconf(_SC_PAGESIZE), n); }
void *pvalloc(size_t n)
{
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    return memalign(page, (n + page - 1) & ~(page - 1));
}

/* A fork child starts with the parent's live objects; its own counts start at zero and its peak at that live set. */
static void child_after_fork(void)
{
    forked = 1;
    allocs = frees = moved = inplace = failed = 0;
    memset(size_log2, 0, sizeof size_log2);
    peak = live;
    peak_bytes = live_bytes;
}

__attribute__((constructor)) static void init(void)
{
    const char *path = getenv("SLOTCENSUS_OUT");
    if (path && strlen(path) < sizeof out_path)
        strcpy(out_path, path);
    resolve();
    pthread_atfork(NULL, NULL, child_after_fork);
}

__attribute__((destructor)) static void report(void)
{
    const char *path = out_path;
    if (!*path)
        return;
    char cmd[1024] = "?";
    int fd = open("/proc/self/cmdline", O_RDONLY);
    if (fd >= 0) {
        ssize_t n = read(fd, cmd, sizeof cmd - 1);
        n = n > 0 ? n : 0;
        while (n > 0 && cmd[n - 1] == 0) --n;   /* the trailing NUL */
        cmd[n] = 0;
        for (ssize_t i = 0; i < n; ++i)
            cmd[i] = cmd[i] == 0 ? ' ' : cmd[i] == '"' || cmd[i] == '\\' || (unsigned char)cmd[i] < 0x20 ? '_' : cmd[i];
        close(fd);
    }
    char line[4096];
    int len = snprintf(line, sizeof line,
                       "{\"pid\":%d,\"ppid\":%d,\"cmdline\":\"%s\",\"forked\":%d,\"peak_live\":%lld,\"end_live\":%lld,"
                       "\"allocs\":%lld,\"frees\":%lld,\"realloc_moved\":%lld,\"realloc_inplace\":%lld,\"failed\":%lld,"
                       "\"peak_live_bytes\":%lld,\"size_log2\":[",
                       (int)getpid(), (int)getppid(), cmd, forked, peak, live, allocs, frees, moved, inplace,
                       failed, peak_bytes);
    for (int b = 0; b < 64 && len < (int)sizeof line - 32; ++b)
        len += snprintf(line + len, sizeof line - len, "%s%lld", b ? "," : "", size_log2[b]);
    len += snprintf(line + len, sizeof line - len, "]}\n");
    fd = open(path, O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd >= 0) {
        if (write(fd, line, (size_t)len) < 0) { /* nothing to do */ }
        close(fd);
    }
}
