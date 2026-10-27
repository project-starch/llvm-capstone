/* census.h: the counting rules slotcensus.c, ffcensus.c and slotins.c share, so the three tracers count alike.
 *
 * Exact at every event: every successful allocation and free updates one live counter, and the peak is the running
 * maximum of that counter (compare-and-swap), never a sample. Rules:
 *   allocation        a successful call is one allocation and one live object, a failed one (NULL) a failure;
 *                     malloc(0) returns a unique pointer, so it is an object
 *   free              of a non-null pointer: one free, one live object less
 *   realloc(NULL, n)  an allocation (the caller counts it as one)
 *   realloc(p, n)     a NULL result with n == 0 frees p (glibc), with n > 0 fails and leaves p alone; a result at the
 *                     old address is in place (the live count unchanged); elsewhere a move: one allocation and one
 *                     free, the live count unchanged
 *   fork              the child inherits the parent's live objects: its counts start at zero and its peaks at that
 *                     live set
 * Bytes are the allocator's usable sizes (0 where a tracer cannot ask), so live bytes include its rounding. Sizes are
 * the requested ones, counted per bucket floor(log2(size)) (bucket 0 holds 0 and 1). */
#ifndef SLOTCENSUS_CENSUS_H
#define SLOTCENSUS_CENSUS_H

#include <stddef.h>
#include <stdint.h>

/* the layout ffcensus.c maps into its per-process file and ffcensus-dump.py reads: keep the order */
struct census_counts {
    int64_t live, peak, allocs, frees, moved, inplace, failed, live_bytes, peak_bytes;
    int64_t size_log2[64];
};

static inline void census_raise_max(int64_t *max, int64_t v)
{
    int64_t old = __atomic_load_n(max, __ATOMIC_RELAXED);
    while (v > old && !__atomic_compare_exchange_n(max, &old, v, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

static inline void census_add(int64_t *counter, int64_t n) { __atomic_add_fetch(counter, n, __ATOMIC_RELAXED); }

static inline void census_size(struct census_counts *c, size_t requested)
{
    unsigned b = requested > 1 ? 63 - (unsigned)__builtin_clzll((unsigned long long)requested) : 0;
    census_add(&c->size_log2[b], 1);
}

/* the result of an allocating call: p non-null on success */
static inline void census_alloc(struct census_counts *c, const void *p, size_t requested, int64_t usable)
{
    if (!p) {
        census_add(&c->failed, 1);
        return;
    }
    census_add(&c->allocs, 1);
    census_size(c, requested);
    census_raise_max(&c->peak, __atomic_add_fetch(&c->live, 1, __ATOMIC_RELAXED));
    census_raise_max(&c->peak_bytes, __atomic_add_fetch(&c->live_bytes, usable, __ATOMIC_RELAXED));
}

/* the free of a non-null pointer with this usable size */
static inline void census_free(struct census_counts *c, int64_t usable)
{
    census_add(&c->frees, 1);
    census_add(&c->live, -1);
    census_add(&c->live_bytes, -usable);
}

/* realloc of a non-null old object to n bytes: p the result, was and now the old and new usable sizes */
static inline void census_realloc(struct census_counts *c, const void *old, const void *p, size_t n, int64_t was,
                                  int64_t now)
{
    if (!p) {
        if (n == 0)
            census_free(c, was);
        else
            census_add(&c->failed, 1);
        return;
    }
    census_size(c, n);
    if (p == old) {
        census_add(&c->inplace, 1);
    } else {
        census_add(&c->moved, 1);
        census_add(&c->allocs, 1);
        census_add(&c->frees, 1);
    }
    census_raise_max(&c->peak_bytes, __atomic_add_fetch(&c->live_bytes, now - was, __ATOMIC_RELAXED));
}

static inline void census_after_fork(struct census_counts *c)
{
    c->allocs = c->frees = c->moved = c->inplace = c->failed = 0;
    for (int b = 0; b < 64; ++b)
        c->size_log2[b] = 0;
    c->peak = c->live;
    c->peak_bytes = c->live_bytes;
}

/* /proc/self/cmdline as read (n bytes) into one JSON-safe line: arguments joined by spaces, quotes, backslashes
 * and control characters as '_'; returns its length (s has room for n + 1 bytes) */
static inline long census_cmdline(char *s, long n)
{
    n = n > 0 ? n : 0;
    while (n > 0 && s[n - 1] == 0)   /* the trailing NUL */
        --n;
    for (long i = 0; i < n; ++i)
        s[i] = s[i] == 0 ? ' ' : s[i] == '"' || s[i] == '\\' || (unsigned char)s[i] < 0x20 ? '_' : s[i];
    s[n] = 0;
    return n;
}

#endif
