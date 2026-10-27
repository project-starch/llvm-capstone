/* slotcensus: an LD_PRELOAD census of simultaneously live heap objects, for sizing lifetime-id slots.
 *
 * Counted entry points: malloc, calloc, realloc, reallocarray, free, posix_memalign, aligned_alloc, memalign, valloc,
 * pvalloc (C++ operator new/delete reach malloc and free through libstdc++), by the rules in census.h.
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

#include "census.h"

static void *(*r_malloc)(size_t);
static void *(*r_calloc)(size_t, size_t);
static void *(*r_realloc)(void *, size_t);
static void (*r_free)(void *);
static int (*r_posix_memalign)(void **, size_t, size_t);
static void *(*r_aligned_alloc)(size_t, size_t);
static void *(*r_memalign)(size_t, size_t);

static struct census_counts counts;
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

static void note_alloc(void *p, size_t requested)
{
    census_alloc(&counts, p, requested, p ? (int64_t)malloc_usable_size(p) : 0);
}

static void note_free(void *p)
{
    if (p && !in_boot(p))
        census_free(&counts, (int64_t)malloc_usable_size(p));
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
    int64_t was = (int64_t)malloc_usable_size(o);
    void *p = r_realloc(o, n);
    census_realloc(&counts, o, p, n, was, p ? (int64_t)malloc_usable_size(p) : 0);
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

static void child_after_fork(void)
{
    forked = 1;
    census_after_fork(&counts);
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
        census_cmdline(cmd, read(fd, cmd, sizeof cmd - 1));
        close(fd);
    }
    char line[4096];
    int len = snprintf(line, sizeof line,
                       "{\"pid\":%d,\"ppid\":%d,\"cmdline\":\"%s\",\"forked\":%d,\"peak_live\":%lld,\"end_live\":%lld,"
                       "\"allocs\":%lld,\"frees\":%lld,\"realloc_moved\":%lld,\"realloc_inplace\":%lld,\"failed\":%lld,"
                       "\"peak_live_bytes\":%lld,\"size_log2\":[",
                       (int)getpid(), (int)getppid(), cmd, forked, (long long)counts.peak, (long long)counts.live,
                       (long long)counts.allocs, (long long)counts.frees, (long long)counts.moved,
                       (long long)counts.inplace, (long long)counts.failed, (long long)counts.peak_bytes);
    for (int b = 0; b < 64 && len < (int)sizeof line - 32; ++b)
        len += snprintf(line + len, sizeof line - len, "%s%lld", b ? "," : "", (long long)counts.size_log2[b]);
    len += snprintf(line + len, sizeof line - len, "]}\n");
    fd = open(path, O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd >= 0) {
        if (write(fd, line, (size_t)len) < 0) { /* nothing to do */ }
        close(fd);
    }
}
