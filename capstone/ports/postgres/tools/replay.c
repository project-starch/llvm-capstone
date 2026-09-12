/* Replay a recorded PostgreSQL workload against PostgreSQL's memory manager,
 * on the host.
 *
 * The manager is the seven files of src/backend/utils/mmgr, unchanged, linked
 * against pg_stubs.c and nothing else of the backend. The workload is a trace
 * recorded inside a real backend under pgbench: every call to the context
 * interface, in order, naming contexts and objects by identity.
 *
 * The loop is replay_core.inc, shared with the domain driver. This file is the
 * host's half of it: it maps a file and prints with printf. It answers the
 * question that has to be answered before any of that, and it answers it
 * exactly: does the manager ask the level below for the same thing when it is
 * driven by the same sequence.
 *
 *     replay <trace> [--quiet]
 */
#include "postgres.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static const char *me = "replay";

static void die(const char *what)
{
    fprintf(stderr, "%s: %s\n", me, what);
    exit(2);
}
static void die_at(unsigned long i, const char *what, unsigned long id)
{
    fprintf(stderr, "%s: record %lu: %s (%lu)\n", me, i, what, id);
    exit(2);
}

/* ---- level 0, counted ---------------------------------------------------
 * The manager takes its blocks from malloc. Interposing here counts them
 * without touching the manager, which is the point: the arm that is measured
 * has to be the manager as it ships.
 */
extern void *__libc_malloc(size_t);
extern void __libc_free(void *);
extern void *__libc_realloc(void *, size_t);

static unsigned long blk_alloc, blk_free, blk_realloc, blk_live, blk_peak;

void *malloc(size_t n)
{
    void *p = __libc_malloc(n);
    if (p) { blk_alloc++; if (++blk_live > blk_peak) blk_peak = blk_live; }
    return p;
}
void free(void *p)
{
    if (p) { blk_free++; blk_live--; }
    __libc_free(p);
}
void *realloc(void *p, size_t n)
{
    blk_realloc++;
    return __libc_realloc(p, n);
}

/* The identity tables are this program's memory and not the manager's, so they
   come from the level below without passing the counters. */
static void *replay_alloc(size_t n)
{
    void *p = __libc_malloc(n);
    if (p) memset(p, 0, n);
    return p;
}

#define REPLAY_DIE(msg) die(msg)
#define REPLAY_DIE_AT(i, msg, id) die_at((i), (msg), (id))
#define REPLAY_ALLOC(n) replay_alloc(n)
#include "replay_core.inc"

int
main(int argc, char **argv)
{
    int quiet = 0;

    for (int i = 2; i < argc; i++)
        if (strcmp(argv[i], "--quiet") == 0)
            quiet = 1;
    if (argc < 2)
        die("usage: replay <trace> [--quiet]");

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0)
        die("cannot open the trace");
    struct stat st;
    if (fstat(fd, &st) != 0)
        die("cannot size the trace");
    char *base = mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED)
        die("cannot map the trace");

    struct a11_head *h = (struct a11_head *) base;
    if (memcmp(h->magic, A11_MAGIC, 8) != 0)
        die("not a trace");
    if (h->version != A11_VERSION || h->recsize != sizeof(struct a11_rec)
        || h->endian != A11_ENDIAN)
        die("a trace this reader does not agree with");
    if (h->ppid != 0)
        die("this trace continues another; flatten it first");

    struct a11_rec *r = (struct a11_rec *) (base + sizeof *h);
    unsigned long n = ((size_t) st.st_size - sizeof *h) / sizeof *r;

    struct replay_counts c;
    memset(&c, 0, sizeof c);
    blk_alloc = blk_free = blk_live = blk_peak = blk_realloc = 0;
    replay_run(r, n, &c);

    /* Read the counters before anything is printed: the first printf takes a
       buffer from the level below, and that block is this program's, not the
       manager's. */
    unsigned long got_alloc = blk_alloc, got_free = blk_free,
                  got_realloc = blk_realloc, got_peak = blk_peak;

    if (!quiet) {
        printf("| what | the trace asked for |\n|---|---:|\n");
        printf("| create | %lu |\n", c.create);
        printf("| alloc | %lu |\n", c.alloc);
        printf("| free | %lu |\n", c.free);
        printf("| realloc | %lu |\n", c.realloc);
        printf("| reset | %lu |\n", c.reset);
        printf("| delete | %lu |\n", c.delete);
#ifdef REPLAY_CHECK_DATA
        printf("| objects whose contents were read back | %lu |\n", c.checked);
#endif
        printf("\n| what the manager asked of the level below | in the backend | here |\n");
        printf("|---|---:|---:|\n");
        printf("| blocks taken | %lu | %lu |\n", c.was_alloc, got_alloc);
        printf("| blocks given back | %lu | %lu |\n", c.was_free, got_free);
        printf("| blocks grown or moved | %lu | %lu |\n", c.was_realloc, got_realloc);
        printf("| blocks held at once, most | %lu | %lu |\n", c.was_peak, got_peak);
    }
    if (!c.have_was) {
        fprintf(stderr, "%s: the trace does not say what the manager took, "
                "so there is nothing to be right about\n", me);
        return 1;
    }
    if (c.was_alloc != got_alloc || c.was_free != got_free || c.was_peak != got_peak) {
        fprintf(stderr, "%s: the manager did not ask the level below for the "
                "same thing: %lu/%lu/%lu in the backend against %lu/%lu/%lu here\n",
                me, c.was_alloc, c.was_free, c.was_peak, got_alloc, got_free, got_peak);
        return 1;
    }
    if (!quiet)
        printf("\nthe same blocks, in the same number, held the same way\n");
    return 0;
}
