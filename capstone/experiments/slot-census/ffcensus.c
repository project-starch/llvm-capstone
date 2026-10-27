/* ffcensus: slotcensus for Firefox, as a replace-malloc library (Nightly builds; on Linux firefox-bin takes the weak
 * symbol replace_init from a preloaded library, which is all this one exports).
 *
 * Firefox allocates through its own mozjemalloc, which an LD_PRELOAD malloc tracer cannot see; replace-malloc hands
 * this library mozjemalloc's function table at start-up, and it puts its counting functions in front of the originals:
 * malloc, calloc, realloc, free, memalign, posix_memalign, aligned_alloc, valloc and the moz_arena_* family. The
 * counting is slotcensus.c's, the rules in census.h.
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

#include "census.h"
#include "replace_malloc.h"

#define FFCENSUS_MAGIC 0x53554e4543534646ull   /* "FFCENSUS" */

struct census {   /* the file layout ffcensus-dump.py reads */
    uint64_t magic;
    int32_t pid, ppid, forked, pad;
    struct census_counts counts;
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
        long n = census_cmdline(buf, read(fd, buf, sizeof buf - 1));
        close(fd);
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

static int64_t usable(void *p) { return p ? (int64_t)orig.malloc_usable_size(p) : 0; }

static void note_alloc(void *p, size_t requested) { census_alloc(&c->counts, p, requested, usable(p)); }

static void note_free(void *p)
{
    if (p)
        census_free(&c->counts, usable(p));
}

static void *malloc_counted(size_t n) { void *p = orig.malloc(n); note_alloc(p, n); return p; }
static void *calloc_counted(size_t a, size_t b) { void *p = orig.calloc(a, b); note_alloc(p, a * b); return p; }
static void free_counted(void *p) { note_free(p); orig.free(p); }
static void *memalign_counted(size_t al, size_t n) { void *p = orig.memalign(al, n); note_alloc(p, n); return p; }
static void *aligned_alloc_counted(size_t al, size_t n)
{
    void *p = orig.aligned_alloc(al, n);
    note_alloc(p, n);
    return p;
}
static void *valloc_counted(size_t n) { void *p = orig.valloc(n); note_alloc(p, n); return p; }

static int posix_memalign_counted(void **out, size_t al, size_t n)
{
    int rc = orig.posix_memalign(out, al, n);
    note_alloc(rc == 0 ? *out : NULL, n);
    return rc;
}

static void *realloc_counted(void *o, size_t n)
{
    if (!o) { void *p = orig.realloc(NULL, n); note_alloc(p, n); return p; }
    int64_t was = usable(o);
    void *p = orig.realloc(o, n);
    census_realloc(&c->counts, o, p, n, was, usable(p));
    return p;
}

static void *arena_malloc_counted(arena_id_t a, size_t n)
{
    void *p = orig.moz_arena_malloc(a, n);
    note_alloc(p, n);
    return p;
}
static void arena_free_counted(arena_id_t a, void *p) { note_free(p); orig.moz_arena_free(a, p); }

static void *arena_calloc_counted(arena_id_t a, size_t k, size_t n)
{
    void *p = orig.moz_arena_calloc(a, k, n);
    note_alloc(p, k * n);
    return p;
}

static void *arena_memalign_counted(arena_id_t a, size_t al, size_t n)
{
    void *p = orig.moz_arena_memalign(a, al, n);
    note_alloc(p, n);
    return p;
}

static void *arena_realloc_counted(arena_id_t a, void *o, size_t n)
{
    if (!o) { void *p = orig.moz_arena_realloc(a, NULL, n); note_alloc(p, n); return p; }
    int64_t was = usable(o);
    void *p = orig.moz_arena_realloc(a, o, n);
    census_realloc(&c->counts, o, p, n, was, usable(p));
    return p;
}

/* Arena creation and disposal pass straight through, but must be replaced too: with the moz_arena_* allocation
 * functions replaced and these left alone, Firefox crashed in its first moz_arena_malloc (measured; with both
 * replaced it runs). */
static arena_id_t create_arena_counted(arena_params_t *params) { return orig.moz_create_arena_with_params(params); }
static void dispose_arena_counted(arena_id_t a) { orig.moz_dispose_arena(a); }

/* a fork child starts at the parent's live set, with its own file and counts (as slotcensus.c's child) */
static void child_after_fork(void)
{
    struct census snap = *c;
    snap.forked = 1;
    census_after_fork(&snap.counts);
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
    table->malloc = malloc_counted;
    table->calloc = calloc_counted;
    table->realloc = realloc_counted;
    table->free = free_counted;
    table->memalign = memalign_counted;
    table->posix_memalign = posix_memalign_counted;
    table->aligned_alloc = aligned_alloc_counted;
    table->valloc = valloc_counted;
    table->moz_create_arena_with_params = create_arena_counted;
    table->moz_dispose_arena = dispose_arena_counted;
    table->moz_arena_malloc = arena_malloc_counted;
    table->moz_arena_calloc = arena_calloc_counted;
    table->moz_arena_realloc = arena_realloc_counted;
    table->moz_arena_free = arena_free_counted;
    table->moz_arena_memalign = arena_memalign_counted;
    pthread_atfork(NULL, NULL, child_after_fork);
}
