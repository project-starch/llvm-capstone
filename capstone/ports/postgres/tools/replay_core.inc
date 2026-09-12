/* The replay itself: the loop that turns a trace back into calls.
 *
 * No input and no output. A driver maps or is handed the records, calls
 * replay_run, and reports. The host driver and the domain driver differ in
 * exactly those two things and in nothing else, which is what makes a
 * difference between their results a difference about capabilities.
 *
 * What the includer must provide before including this:
 *
 *   REPLAY_DIE(msg)            what an unusable trace does
 *   REPLAY_DIE_AT(i, msg, id)  what a name that does not resolve does
 *   REPLAY_ALLOC(n)            zeroed memory for the identity tables,
 *                              which are not the manager's blocks
 *
 * Define REPLAY_CHECK_DATA to make the replay check that what it wrote into an
 * object is still there when the trace next mentions it. Counting the calls
 * says the allocator's bookkeeping is consistent; this says the objects hold
 * their contents, which is the contract palloc actually makes and the thing a
 * discipline that revokes and re-hands memory could break without any counter
 * noticing. It costs a third table and a pass over every object's bytes.
 *
 * The moments the trace gives are the free and the realloc, and both are
 * taken. A reset and a delete do not name the objects they take, so there is
 * no moment at which those can be checked, and nothing is swept at the end
 * either: after a reset the table's entries for that context's objects point
 * at memory the discipline has revoked, and reading one would fault rather
 * than report. So the coverage is every object the trace frees or reallocs,
 * which the driver prints, and it is stated rather than implied.
 */
#include "utils/memutils.h"
#include "a11trace.h"

struct replay_counts {
    unsigned long create, alloc, free, realloc, reset, delete;
    unsigned long peak;                 /* objects alive at once, most */
    unsigned long checked;              /* objects whose contents were read back */
    /* what the backend's manager asked of the level below, from the trace */
    unsigned long was_alloc, was_free, was_realloc, was_peak;
    int have_was;
};

/* The oracle, and it stays in: a replay that follows a name it was never given
   is not replaying the recording, and the way that shows without a check is a
   fault somewhere else entirely. */
#define REPLAY_CTX(i, id) \
    (ctx[id] ? ctx[id] : (REPLAY_DIE_AT((i), "no context with this id", (id)), (MemoryContext) 0))
#define REPLAY_PTR(i, id) \
    (ptr[id] ? ptr[id] : (REPLAY_DIE_AT((i), "no object with this id", (id)), (void *) 0))

#ifdef REPLAY_CHECK_DATA
/* A pattern that depends on the object and on the offset, so that a byte from
 * the wrong object or the wrong place in the right one is caught. It is read
 * and written a word at a time and never as a pointer: a capability written
 * here would be revoked with the chunk and the read back would fault rather
 * than mismatch, which says less. */
static unsigned long
replay_word(unsigned long id, unsigned long k)
{
    return (id * 0x9E3779B97F4A7C15UL) ^ (k * 0xBF58476D1CE4E5B9UL);
}

static void
replay_fill(void *p, unsigned long id, unsigned long bytes)
{
    unsigned long *w = (unsigned long *) p;
    unsigned long words = bytes / sizeof *w;

    for (unsigned long k = 0; k < words; k++)
        w[k] = replay_word(id, k);
}

/* Returns the index of the first word that differs, or the word count when
 * everything matched. */
static unsigned long
replay_verify(void *p, unsigned long id, unsigned long bytes)
{
    unsigned long *w = (unsigned long *) p;
    unsigned long words = bytes / sizeof *w;

    for (unsigned long k = 0; k < words; k++)
        if (w[k] != replay_word(id, k))
            return k;
    return words;
}
#endif

static void
replay_run(struct a11_rec *r, unsigned long n, struct replay_counts *c)
{
    /* The footer says how many identities were handed out, so the two tables
       are sized exactly once and never grow. A domain has no room for a table
       that doubles. */
    if (n == 0 || r[n - 1].op != A11_END)
        REPLAY_DIE("the trace has no footer: it is truncated");
    unsigned long nctx = r[n - 1].s2 + 2, nptr = r[n - 1].s3 + 2;
    MemoryContext *ctx = REPLAY_ALLOC(nctx * sizeof *ctx);
    void **ptr = REPLAY_ALLOC(nptr * sizeof *ptr);
    if (!ctx || !ptr)
        REPLAY_DIE("no room for the identity tables");
#ifdef REPLAY_CHECK_DATA
    /* How many bytes each object was given, so that the check knows how far to
       read. Four bytes an object, against sixteen for its capability. */
    unsigned int *len = REPLAY_ALLOC(nptr * sizeof *len);

    if (!len)
        REPLAY_DIE("no room for the table the data check needs");
#endif

    unsigned long live = 0;

    for (unsigned long i = 0; i < n; i++) {
        struct a11_rec *e = &r[i];

        switch (e->op) {
            case A11_CREATE_ASET:
            case A11_CREATE_GEN:
            case A11_CREATE_SLAB:
            case A11_CREATE_BUMP: {
                MemoryContext parent = e->aux ? REPLAY_CTX(i, e->aux) : NULL;
                MemoryContext k;

                if (e->op == A11_CREATE_ASET)
                    k = AllocSetContextCreateInternal(parent, "replay", e->s1, e->s2, e->s3);
                else if (e->op == A11_CREATE_GEN)
                    k = GenerationContextCreate(parent, "replay", e->s1, e->s2, e->s3);
                else if (e->op == A11_CREATE_SLAB)
                    k = SlabContextCreate(parent, "replay", e->s1, e->s2);
                else
                    k = BumpContextCreate(parent, "replay", e->s1, e->s2, e->s3);
                ctx[e->ctx] = k;
                /* The first root is the one the backend made first, and the
                   manager reads these two globals on paths that report. */
                if (!parent && TopMemoryContext == NULL)
                    TopMemoryContext = CurrentMemoryContext = k;
                c->create++;
                break;
            }
            case A11_ALLOC:
                ptr[e->ptr] = MemoryContextAlloc(REPLAY_CTX(i, e->ctx), e->s1);
#ifdef REPLAY_CHECK_DATA
                len[e->ptr] = (unsigned int) e->s1;
                if (ptr[e->ptr])
                    replay_fill(ptr[e->ptr], e->ptr, e->s1);
#endif
                c->alloc++;
                if (++live > c->peak) c->peak = live;
                break;
            case A11_FREE:
#ifdef REPLAY_CHECK_DATA
                if (replay_verify(REPLAY_PTR(i, e->ptr), e->ptr, len[e->ptr])
                    != len[e->ptr] / sizeof(unsigned long))
                    REPLAY_DIE_AT(i, "an object did not hold what was written into it",
                                  e->ptr);
                c->checked++;
#endif
                pfree(REPLAY_PTR(i, e->ptr));
                ptr[e->ptr] = NULL;
                c->free++; live--;
                break;
            case A11_REALLOC: {
#ifdef REPLAY_CHECK_DATA
                unsigned long was = len[e->ptr];

                if (replay_verify(REPLAY_PTR(i, e->ptr), e->ptr, was)
                    != was / sizeof(unsigned long))
                    REPLAY_DIE_AT(i, "an object did not hold what was written into it",
                                  e->ptr);
#endif
                void *q = repalloc(REPLAY_PTR(i, e->ptr), e->s1);
#ifdef REPLAY_CHECK_DATA
                /* repalloc promises the old contents, so they are checked
                   after the move and not only before it. Under the Sublet port
                   a large chunk moves by copy where upstream grew its block in
                   place, and this is the check that says the copy was right
                   and kept its tags. */
                unsigned long kept = was < e->s1 ? was : e->s1;

                if (q && replay_verify(q, e->ptr, kept)
                    != kept / sizeof(unsigned long))
                    REPLAY_DIE_AT(i, "a realloc did not keep the old contents",
                                  e->ptr);
                if (q && e->aux) {
                    len[e->aux] = (unsigned int) e->s1;
                    replay_fill(q, e->aux, e->s1);
                }
                c->checked += 2;                 /* before the move and after */
#endif
                ptr[e->ptr] = NULL;
                if (e->aux) ptr[e->aux] = q;
                c->realloc++;
                break;
            }
            case A11_RESET:
                MemoryContextReset(REPLAY_CTX(i, e->ctx));
                c->reset++;
                break;
            case A11_DELETE:
                MemoryContextDelete(REPLAY_CTX(i, e->ctx));
                ctx[e->ctx] = NULL;
                c->delete++;
                break;
            case A11_BLOCKS:
                c->was_alloc = e->s1; c->was_free = e->s2;
                c->was_realloc = e->s3; c->was_peak = e->aux;
                c->have_was = 1;
                break;
            case A11_END:
                break;
            default:
                REPLAY_DIE("a record this reader does not know");
        }
    }
}
