/* The level below PostgreSQL's memory manager, in a domain.
 *
 * The manager takes blocks with malloc and gives them back with free. On the
 * host that is glibc. A domain has no libc, so this is it: first fit over one
 * region, with a free list and coalescing, which is enough to be a real
 * allocator rather than a bump pointer that never reuses.
 *
 * It has to be real, because the measurement is about what the manager costs
 * ABOVE the level below, and a level below that never reuses would make every
 * block fresh and the comparison meaningless.
 *
 * It is deliberately simple, and it is not the subject: 194 blocks alive at
 * once in the recorded workload, almost all of them 8 KiB. If the measurement
 * ever turns on what level 0 does, memsys5 is in this repository, already runs
 * freestanding in a domain, and would replace this file so that both ports
 * stand on the same allocator.
 *
 * The region comes from the host, and pg_level0_init names it once.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* A header before every block. `size` is the payload, `prev` walks backwards
   so a free can coalesce with what is behind it without a scan. */
struct hdr {
    size_t size;
    size_t prevsize;            /* 0 when the block behind is in use */
    struct hdr *next_free;
    struct hdr *prev_free;
};

#define ALIGN 16                /* a capability's alignment */
#define HDRSZ ((sizeof(struct hdr) + ALIGN - 1) & ~(size_t) (ALIGN - 1))
#define ROUND(n) (((n) + ALIGN - 1) & ~(size_t) (ALIGN - 1))
#define PAYLOAD(h) ((void *) ((char *) (h) + HDRSZ))
#define HEADER(p) ((struct hdr *) ((char *) (p) - HDRSZ))
#define NEXT(h) ((struct hdr *) ((char *) (h) + HDRSZ + (h)->size))

static char *arena_lo, *arena_hi;
static struct hdr *free_head;
static int in_use;              /* blocks the manager holds */

unsigned long pg_level0_taken, pg_level0_given, pg_level0_grown;
unsigned long pg_level0_live, pg_level0_peak;

static void
unlink_free(struct hdr *h)
{
    if (h->prev_free)
        h->prev_free->next_free = h->next_free;
    else
        free_head = h->next_free;
    if (h->next_free)
        h->next_free->prev_free = h->prev_free;
    h->next_free = h->prev_free = NULL;
}

static void
link_free(struct hdr *h)
{
    h->prev_free = NULL;
    h->next_free = free_head;
    if (free_head)
        free_head->prev_free = h;
    free_head = h;
}

void
pg_level0_init(void *region, size_t bytes)
{
    /* Align by moving the pointer, never by masking an address and casting it
       back: that would discard the capability and the first dereference would
       fault. The compiler says so (-Wcapstone-pointer-roundtrip), and it said
       so about this line. */
    /* Reading the address is one way and safe; it is the way back that would
       lose the tag. The compiler warns that uintptr_t is narrower than a
       pointer here, which is true and is the point: the low bits are all the
       alignment question needs. */
    size_t skew = (size_t) ((uintptr_t) region & (uintptr_t) (ALIGN - 1));
    char *base = (char *) region + (skew ? ALIGN - skew : 0);
    size_t usable = bytes - (size_t) (base - (char *) region);

    arena_lo = base;
    arena_hi = base + (usable & ~(size_t) (ALIGN - 1));

    struct hdr *h = (struct hdr *) arena_lo;
    h->size = (size_t) (arena_hi - arena_lo) - HDRSZ;
    h->prevsize = 0;
    h->next_free = h->prev_free = NULL;
    free_head = h;
    in_use = 0;
    pg_level0_taken = pg_level0_given = pg_level0_grown = 0;
    pg_level0_live = pg_level0_peak = 0;
}

void *
malloc(size_t want)
{
    if (want == 0)
        want = 1;
    size_t need = ROUND(want);

    for (struct hdr *h = free_head; h; h = h->next_free) {
        if (h->size < need)
            continue;
        unlink_free(h);
        /* split, when what is left can hold a header and something */
        if (h->size >= need + HDRSZ + ALIGN) {
            struct hdr *rest = (struct hdr *) ((char *) h + HDRSZ + need);
            rest->size = h->size - need - HDRSZ;
            rest->prevsize = 0;
            rest->next_free = rest->prev_free = NULL;
            h->size = need;
            struct hdr *after = NEXT(rest);
            if ((char *) after < arena_hi)
                after->prevsize = 0;
            link_free(rest);
        }
        struct hdr *after = NEXT(h);
        if ((char *) after < arena_hi)
            after->prevsize = 0;         /* in use, so it may not coalesce back */
        in_use++;
        pg_level0_taken++;
        if (++pg_level0_live > pg_level0_peak)
            pg_level0_peak = pg_level0_live;
        return PAYLOAD(h);
    }
    return NULL;
}

void
free(void *p)
{
    if (!p)
        return;
    struct hdr *h = HEADER(p);

    pg_level0_given++;
    pg_level0_live--;
    in_use--;

    /* forward */
    struct hdr *after = NEXT(h);
    if ((char *) after < arena_hi && after->prevsize == 0 && after->next_free) {
        /* after is free: it is on the list, so take it off and swallow it */
        unlink_free(after);
        h->size += HDRSZ + after->size;
    } else if ((char *) after < arena_hi) {
        after->prevsize = h->size;       /* free, so it may coalesce back */
    }
    /* backwards */
    if (h->prevsize) {
        struct hdr *before = (struct hdr *) ((char *) h - HDRSZ - h->prevsize);
        unlink_free(before);
        before->size += HDRSZ + h->size;
        h = before;
    }
    struct hdr *tail = NEXT(h);
    if ((char *) tail < arena_hi)
        tail->prevsize = h->size;
    link_free(h);
}

void *
realloc(void *p, size_t want)
{
    if (!p)
        return malloc(want);
    if (want == 0) {
        free(p);
        return NULL;
    }
    struct hdr *h = HEADER(p);
    size_t need = ROUND(want);

    pg_level0_grown++;
    if (h->size >= need)
        return p;                        /* in place, and the caller keeps it */

    void *q = malloc(want);
    if (!q)
        return NULL;
    /* memcpy and not a byte loop, and this cost a run to learn: a byte loop
       copies a pointer's address bits and drops its out-of-band tag, so every
       pointer in what was copied comes back untagged. The memory manager keeps
       its block list in the block it is reallocating, so the first store
       through a copied prev faulted with an unexpected operand type. The
       repository's freestanding memcpy copies the aligned middle one
       capability at a time and keeps the tags. */
    memcpy(q, p, h->size);
    free(p);
    return q;
}

void *
calloc(size_t k, size_t n)
{
    size_t bytes = k * n;
    void *p = malloc(bytes);

    if (p)
        memset(p, 0, bytes);
    return p;
}

int pg_level0_in_use(void) { return in_use; }
