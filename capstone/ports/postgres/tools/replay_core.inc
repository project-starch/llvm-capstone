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
 *   REPLAY_ALLOC(n)            zeroed memory for the two identity tables,
 *                              which are not the manager's blocks
 */
#include "utils/memutils.h"
#include "a11trace.h"

struct replay_counts {
    unsigned long create, alloc, free, realloc, reset, delete;
    unsigned long peak;                 /* objects alive at once, most */
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
                c->alloc++;
                if (++live > c->peak) c->peak = live;
                break;
            case A11_FREE:
                pfree(REPLAY_PTR(i, e->ptr));
                ptr[e->ptr] = NULL;
                c->free++; live--;
                break;
            case A11_REALLOC: {
                void *q = repalloc(REPLAY_PTR(i, e->ptr), e->s1);
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
