/* The seam's metadata: fixed slabs of records, and one chained hash over the payload base.
 *
 * WHY SLABS AND NOT AN ARRAY. Each record holds three sublet_cap slots, and a slot is a PLACE:
 * a linear capability moves when it is copied, so a record that is memcpy'd or realloc'd takes
 * its capabilities with it in a way the hardware does not agree to. The records therefore never
 * move. Slabs are allocated from musl (the level below PHP's allocator, not PHP's), chained,
 * and never freed until the process ends; a freed record goes on a free list inside its slab.
 *
 * WHY ONE KEY, AND WHY THAT ONE. The index is keyed on the PAYLOAD BASE -- B + 48, the pointer
 * PHP hands its callers. Every lookup derives it:
 *   _efree / _erealloc / _mem_block_check   ZEND_CAP_HEADER(ptr)    key = cursor(ptr)
 *   _emalloc, both paths                    ZEND_CAP_PAYLOAD(p, n)  key = cursor(p) + 48
 * so neither direction needs a capability that spans the boundary, which is the whole point of
 * the seam.
 *
 * WHAT THIS DELIBERATELY DOES NOT COPY. The physical port's slot table is a linear scan with
 * `base == 0` marking a free slot, and zend_capstone_alloc.h:340-353 records that sentinel
 * producing a FALSE POSITIVE -- a live lookup matching a free slot. Here a free record is
 * marked by `live == 0` and is not in any chain, and `zend_cap_find` returns only live records.
 *
 * SIZES are -D-overridable because the right ones are unknown until PHP has been measured:
 * every zval and every string reaches _emalloc in this build (ZEND_ENABLE_FAST_CACHE 0), so the
 * high-water is a measurement, not a prediction.
 */
#include "zend-cap-port.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ZEND_CAP_RECORDS_PER_SLAB
#define ZEND_CAP_RECORDS_PER_SLAB 4096UL
#endif
#ifndef ZEND_CAP_MAX_SLABS
#define ZEND_CAP_MAX_SLABS 256UL        /* 256 * 4096 = 1,048,576 live blocks */
#endif
#ifndef ZEND_CAP_BUCKETS
#define ZEND_CAP_BUCKETS 16384UL        /* a power of two; see bucket() */
#endif

struct zend_cap_slab {
    struct zend_cap_slab *next;
    unsigned long used;                       /* records handed out of this slab */
    struct zend_cap_block records[ZEND_CAP_RECORDS_PER_SLAB];
};

static struct zend_cap_slab *slabs;
static struct zend_cap_block *free_records;   /* chained through ->next, live == 0 */
static struct zend_cap_block *buckets[ZEND_CAP_BUCKETS];
static unsigned long records_live, records_peak, slab_count, lookups, collisions;

/* heap.c's hash (heap.c:23-24), on the same kind of key: a 16-aligned heap address. The shift
 * drops the bits that are always zero, the multiply is the usual 64-bit golden-ratio mix. */
static unsigned long bucket(unsigned long key)
{
    return ((key >> 4) * 11400714819323198485UL) >> (64 - 14);   /* 2^14 == ZEND_CAP_BUCKETS */
}

static struct zend_cap_slab *slab_new(void)
{
    struct zend_cap_slab *s;
    if (slab_count >= ZEND_CAP_MAX_SLABS) { return (struct zend_cap_slab *)0; }
    /* aligned_alloc, not malloc: the records hold capabilities, and a slot that is not
     * 16-aligned cannot be stored through. 16 is also what musl would give for this size, but
     * asking is cheaper than depending on it. */
    s = (struct zend_cap_slab *) aligned_alloc(ZEND_CAP_GRAIN, sizeof *s);
    if (!s) { return (struct zend_cap_slab *)0; }
    memset(s, 0, sizeof *s);
    s->next = slabs;
    slabs = s;
    ++slab_count;
    return s;
}

struct zend_cap_block *zend_cap_record_new(void)
{
    struct zend_cap_block *r;
    if (free_records) {
        r = free_records;
        free_records = r->next;
    } else {
        struct zend_cap_slab *s = slabs;
        if (!s || s->used == ZEND_CAP_RECORDS_PER_SLAB) {
            s = slab_new();
            if (!s) {
                zend_cap_fail(ZEND_CAP_FAIL_RECORDS, "metadata heap exhausted",
                              records_live, slab_count);
                return (struct zend_cap_block *)0;
            }
        }
        r = &s->records[s->used++];
    }
    /* The slots must be EMPTY, not merely zeroed C memory: a reused record whose slot still
     * held something would have the next sublet_take operate on a stranger's region. */
    sublet_clear(&r->senior);
    sublet_clear(&r->header);
    sublet_clear(&r->payload);
    r->alias = (void *)0;
    r->header_alias = (void *)0;
    r->next = (struct zend_cap_block *)0;
    r->key = r->base = r->bytes = r->size = 0;
    r->issued = 0;
    r->live = 1;
    if (++records_live > records_peak) { records_peak = records_live; }
    return r;
}

void zend_cap_record_free(struct zend_cap_block *r)
{
    if (!r || !r->live) { return; }
    sublet_clear(&r->senior);
    sublet_clear(&r->header);
    sublet_clear(&r->payload);
    r->alias = (void *)0;
    r->header_alias = (void *)0;
    r->live = 0;
    r->key = 0;
    r->next = free_records;
    free_records = r;
    --records_live;
}

void zend_cap_index(struct zend_cap_block *r)
{
    unsigned long b = bucket(r->key);
    if (buckets[b]) { ++collisions; }
    r->next = buckets[b];
    buckets[b] = r;
}

void zend_cap_unindex(struct zend_cap_block *r)
{
    struct zend_cap_block **p = &buckets[bucket(r->key)];
    while (*p && *p != r) { p = &(*p)->next; }
    if (!*p) {
        /* Asked to unindex something that is not in its own chain: the key moved under us, or
         * the record was already removed. Either way the table no longer describes the heap,
         * and continuing would hand out a stale capability. */
        zend_cap_fail(ZEND_CAP_FAIL_UNKNOWN, "unindex: record not in its chain", r->key, 0);
        return;
    }
    *p = r->next;
    r->next = (struct zend_cap_block *)0;
}

struct zend_cap_block *zend_cap_find(unsigned long key)
{
    struct zend_cap_block *r;
    ++lookups;
    for (r = buckets[bucket(key)]; r; r = r->next) {
        if (r->live && r->key == key) { return r; }
    }
    return (struct zend_cap_block *)0;
}

void zend_cap_backing_stats(unsigned long out[4])
{
    out[0] = records_live;
    out[1] = records_peak;
    out[2] = slab_count;
    out[3] = lookups;
}
