/* Replay a recorded PostgreSQL workload against PostgreSQL's memory manager.
 *
 * The manager is the seven files of src/backend/utils/mmgr, unchanged, linked
 * against pg_stubs.c and nothing else of the backend. The workload is a trace
 * recorded inside a real backend under pgbench: every call to the context
 * interface, in order, naming contexts and objects by identity.
 *
 * This is the program a domain runs. On the host it answers the only question
 * that has to be answered before any of that: does the manager make the same
 * calls to the level below when it is driven by the same sequence. The blocks
 * it asks malloc for are counted here, so the answer is a number that can be
 * held against the one measured inside the backend.
 *
 *     replay <trace> [--quiet]
 */
#include "postgres.h"
#include "utils/memutils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "a11trace.h"

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

/* ---- the trace ----------------------------------------------------------- */
static const char *me = "replay";

static void die(const char *what)
{
    fprintf(stderr, "%s: %s\n", me, what);
    exit(2);
}

/* The oracle, and it stays in: a replay that follows a name it was never
   given is not replaying the recording, and the way that shows without a
   check is a fault somewhere else entirely. */
static void die_at(unsigned long i, const char *what, unsigned long id)
{
    fprintf(stderr, "%s: record %lu: %s (%lu)\n", me, i, what, id);
    exit(2);
}
#define CTX(i, id) (ctx[id] ? ctx[id] : (die_at((i), "no context with this id", (id)), (MemoryContext) 0))
#define PTR(i, id) (ptr[id] ? ptr[id] : (die_at((i), "no object with this id", (id)), (void *) 0))

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
    if (n == 0 || r[n - 1].op != A11_END)
        die("the trace has no footer: it is truncated");

    /* The footer says how many identities were handed out, so the two tables
       are sized exactly once and never grow. A domain has no room for a table
       that doubles. */
    unsigned long nctx = r[n - 1].s2 + 2, nptr = r[n - 1].s3 + 2;
    MemoryContext *ctx = calloc(nctx, sizeof *ctx);
    void **ptr = calloc(nptr, sizeof *ptr);
    if (!ctx || !ptr)
        die("no room for the identity tables");
    /* calloc and the tables themselves are not the manager's blocks */
    blk_alloc = blk_free = blk_live = blk_peak = 0;

    unsigned long n_alloc = 0, n_free = 0, n_realloc = 0, n_reset = 0,
                  n_delete = 0, n_create = 0, live = 0, peak = 0;
    /* what the manager took from the level below inside the backend, recorded
       there, so this run can be held against it rather than merely reported */
    unsigned long was_alloc = 0, was_free = 0, was_realloc = 0, was_peak = 0;
    int have_was = 0;

    for (unsigned long i = 0; i < n; i++) {
        struct a11_rec *e = &r[i];

        switch (e->op) {
            case A11_CREATE_ASET:
            case A11_CREATE_GEN:
            case A11_CREATE_SLAB:
            case A11_CREATE_BUMP: {
                MemoryContext parent = e->aux ? CTX(i, e->aux) : NULL;
                MemoryContext c;

                if (e->op == A11_CREATE_ASET)
                    c = AllocSetContextCreateInternal(parent, "replay", e->s1, e->s2, e->s3);
                else if (e->op == A11_CREATE_GEN)
                    c = GenerationContextCreate(parent, "replay", e->s1, e->s2, e->s3);
                else if (e->op == A11_CREATE_SLAB)
                    c = SlabContextCreate(parent, "replay", e->s1, e->s2);
                else
                    c = BumpContextCreate(parent, "replay", e->s1, e->s2, e->s3);
                ctx[e->ctx] = c;
                /* The first root is the one the backend made first, and the
                   manager reads these two globals on paths that report. */
                if (!parent && TopMemoryContext == NULL)
                    TopMemoryContext = CurrentMemoryContext = c;
                n_create++;
                break;
            }
            case A11_ALLOC:
                ptr[e->ptr] = MemoryContextAlloc(CTX(i, e->ctx), e->s1);
                n_alloc++;
                if (++live > peak) peak = live;
                break;
            case A11_FREE:
                pfree(PTR(i, e->ptr));
                ptr[e->ptr] = NULL;
                n_free++; live--;
                break;
            case A11_REALLOC: {
                void *q = repalloc(PTR(i, e->ptr), e->s1);
                ptr[e->ptr] = NULL;
                if (e->aux) ptr[e->aux] = q;
                n_realloc++;
                break;
            }
            case A11_RESET:
                MemoryContextReset(CTX(i, e->ctx));
                n_reset++;
                break;
            case A11_DELETE:
                MemoryContextDelete(CTX(i, e->ctx));
                ctx[e->ctx] = NULL;
                n_delete++;
                break;
            case A11_BLOCKS:
                was_alloc = e->s1; was_free = e->s2; was_realloc = e->s3;
                was_peak = e->aux; have_was = 1;
                break;
            case A11_END:
                break;
            default:
                die("a record this reader does not know");
        }
    }

    /* Read the counters before anything is printed: the first printf takes a
       buffer from the level below, and that block is this program's, not the
       manager's. */
    unsigned long got_alloc = blk_alloc, got_free = blk_free,
                  got_realloc = blk_realloc, got_peak = blk_peak;

    if (!quiet) {
        printf("| what | the trace asked for |\n|---|---:|\n");
        printf("| create | %lu |\n", n_create);
        printf("| alloc | %lu |\n", n_alloc);
        printf("| free | %lu |\n", n_free);
        printf("| realloc | %lu |\n", n_realloc);
        printf("| reset | %lu |\n", n_reset);
        printf("| delete | %lu |\n", n_delete);
        printf("\n| what the manager asked of the level below | in the backend | here |\n");
        printf("|---|---:|---:|\n");
        printf("| blocks taken | %lu | %lu |\n", was_alloc, got_alloc);
        printf("| blocks given back | %lu | %lu |\n", was_free, got_free);
        printf("| blocks grown or moved | %lu | %lu |\n", was_realloc, got_realloc);
        printf("| blocks held at once, most | %lu | %lu |\n", was_peak, got_peak);
    }
    if (!have_was) {
        fprintf(stderr, "%s: the trace does not say what the manager took, "
                "so there is nothing to be right about\n", me);
        return 1;
    }
    if (was_alloc != got_alloc || was_free != got_free || was_peak != got_peak) {
        fprintf(stderr, "%s: the manager did not ask the level below for the "
                "same thing: %lu/%lu/%lu in the backend against %lu/%lu/%lu here\n",
                me, was_alloc, was_free, was_peak, got_alloc, got_free, got_peak);
        return 1;
    }
    if (!quiet)
        printf("\nthe same blocks, in the same number, held the same way\n");
    return 0;
}
