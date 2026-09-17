/* Exact-size payload classes and out-of-band metadata. All comparison arms
 * share the port and layout; only the lifetime discipline changes at runtime.
 * Mode 0: spatial bounds. Mode 1: backing-allocation revocation. Mode 2: Sublet,
 * also revoking each last return to a pool, before reissuing that same storage.
 */
#include "trace.h"
#define ff_memory_init ff2_memory_init
#include "metadata-memory.c"
#ifdef FFPOOL_DOMAIN
#include "../../sqlite/sublet/sublet.h"
#else
typedef struct { void *c; } sublet_cap;
#endif
struct payload_block {
    sublet_cap region, outer;
    void *full_alias, *alias, *meta;
    uintptr_t address;
    size_t requested, rounded;
    unsigned alive, idle;
};
static struct payload_block payload_blocks[2048];
static sublet_cap remaining;
static unsigned mode, nblocks;
static uintptr_t payload_base;
static size_t payload_capacity, payload_used, init_bytes;

void ff2_set_mode(unsigned value)
{ if (value > 2) ff2_fail(301); mode = value; }
void ff2_payload_init(void *p, size_t n)
{
#ifdef FFPOOL_DOMAIN
    sublet_store(&remaining, p);
    payload_base = sublet_base(&remaining);
    if (sublet_type(&remaining) != 0 || sublet_end(&remaining) - payload_base != n) ff2_fail(302);
#else
    remaining.c = p; payload_base = (uintptr_t)p;
#endif
    payload_capacity = n;
}
static void *spatial_alias(struct payload_block *b)
{
#ifdef FFPOOL_DOMAIN
    void *p;
    __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(%1)\n"
                     ".insn s 0x5b, 0x4, x0, 0(%1)\n"
                     ".insn r 0x5b, 0x1, 0x03, %0, x0, x0\n"
                     : "=&r"(p) : "r"(&b->region) : "memory");
    sublet_stats.delin++;
    return p;
#else
    return b->region.c;
#endif
}
static void *issue(struct payload_block *b)
{
    void *p;
#ifdef FFPOOL_DOMAIN
    p = mode == 2 ? sublet_take(&b->region) : b->full_alias;
    p = __builtin_capstone_cap_shrink(p, b->address, b->address + b->requested);
#else
    p = b->full_alias;
#endif
    b->alias = p; b->idle = 0;
    return p;
}
static struct payload_block *by_address(uintptr_t address)
{
    for (unsigned i = 0; i < nblocks; i++)
        if (payload_blocks[i].alive && payload_blocks[i].address == address) return &payload_blocks[i];
    ff2_fail(303);
}
/* Address equality alone does not authorize free. Compare the tagged current
 * capability representation, including its revocation node, at the base cursor.
 * LCC's validity selector is deliberately not used (unimplemented in this QEMU).
 */
static int same_authority(const void *a, const void *b)
{
#ifdef FFPOOL_DOMAIN
    sublet_cap aa, bb;
    sublet_store(&aa, (void *)a); sublet_store(&bb, (void *)b);
    if (sublet_type(&aa) != 1 || sublet_type(&bb) != 1) return 0;
    const volatile uint64_t *x = (const volatile uint64_t *)&aa;
    const volatile uint64_t *y = (const volatile uint64_t *)&bb;
    return x[0] == y[0] && x[1] == y[1];
#else
    return a == b;
#endif
}
static struct payload_block *by_authority(const void *p)
{
    struct payload_block *b = by_address((uintptr_t)p);
    if (b->idle || !same_authority(p, b->alias)) ff2_fail(304);
    return b;
}
void *ff2_payload_alloc(size_t size)
{
    if (!size || size > payload_capacity || size > SIZE_MAX - 63) return NULL;
    size_t rounded = (size + 63) & ~(size_t)63;
    struct payload_block *b = NULL;
    for (unsigned i = 0; i < nblocks; i++)
        if (!payload_blocks[i].alive && payload_blocks[i].rounded == rounded) { b = &payload_blocks[i]; break; }
    if (!b) {
        if (nblocks == 2048 || rounded > payload_capacity - payload_used) return NULL;
        b = &payload_blocks[nblocks++];
        b->address = payload_base + payload_used; b->rounded = rounded;
#ifdef FFPOOL_DOMAIN
        sublet_carve(&remaining, b->address + rounded, &b->region);
#else
        b->region.c = (unsigned char *)remaining.c + payload_used;
#endif
        payload_used += rounded;
    }
    b->requested = size; b->alive = 1; b->idle = 1;
#ifdef FFPOOL_DOMAIN
    if (mode) sublet_handle(&b->region, &b->outer);
#endif
    if (mode != 2 || !b->full_alias) {
        /* Native mode 2 still has an ordinary pointer; hardware mode 2 never
         * exposes a persistent copy of the parent capability. */
#ifdef FFPOOL_DOMAIN
        if (mode != 2 && !b->full_alias) b->full_alias = spatial_alias(b);
#else
        b->full_alias = spatial_alias(b);
#endif
    }
    return issue(b);
}
void ff2_payload_return(void *p)
{
    struct payload_block *b = by_authority(p);
#ifdef FFPOOL_DOMAIN
    if (mode == 2) {
        unsigned long before = sublet_stats.init;
        sublet_give(&b->region);
        if (sublet_stats.init != before) init_bytes += b->rounded;
        b->alias = NULL;
    }
#endif
    b->idle = 1;
}
void *ff2_payload_issue(uintptr_t address)
{
    struct payload_block *b = by_address(address);
    if (!b->idle) ff2_fail(305);
    return issue(b);
}
void ff2_payload_free(void *p)
{
    if (!p) return;
    struct payload_block *b = by_authority(p);
#ifdef FFPOOL_DOMAIN
    if (mode) {
        unsigned long before = sublet_stats.init;
        sublet_give_to(&b->outer, &b->region);
        if (sublet_stats.init != before) init_bytes += b->rounded;
        b->full_alias = b->alias = NULL;
    }
#endif
    b->alive = 0; b->idle = 1;
}
void *ff2_ref_alloc(size_t size, size_t metadata_size)
{
    void *meta = av_mallocz(metadata_size);
    if (!meta) return NULL;
    void *p = ff2_payload_alloc(size);
    if (!p) { av_free(meta); return NULL; }
    by_authority(p)->meta = meta;
    return meta;
}
static struct payload_block *by_meta(void *meta)
{
    for (unsigned i = 0; i < nblocks; i++)
        if (payload_blocks[i].alive && payload_blocks[i].meta == meta) return &payload_blocks[i];
    ff2_fail(306);
}
void *ff2_ref_meta(const void *p) { return by_authority(p)->meta; }
void *ff2_ref_data(void *meta)
{
    struct payload_block *b = by_meta(meta);
    /* Trusted free-entry callbacks may inspect persistent fields in an idle
     * entry. Give them fresh authority, never revive the application's alias. */
    return b->idle ? issue(b) : b->alias;
}
void *ff2_ref_issue(void *meta)
{ struct payload_block *b = by_meta(meta); return ff2_payload_issue(b->address); }
void ff2_ref_return(void *meta)
{ struct payload_block *b = by_meta(meta); ff2_payload_return(b->alias); }
void ff2_ref_free(void *meta)
{
    if (!meta) return;
    struct payload_block *b = by_meta(meta);
    void *p = b->idle ? issue(b) : b->alias;
    ff2_payload_free(p); b->meta = NULL; av_free(meta);
}
void ff2_memory_report(struct ff2_header *h)
{
    h->metadata_used = ff_memory_used(); h->payload_used = payload_used;
#ifdef FFPOOL_DOMAIN
    h->split = sublet_stats.split; h->mrev = sublet_stats.mrev;
    h->delin = sublet_stats.delin; h->revoke = sublet_stats.revoke;
    h->init = sublet_stats.init; h->init_bytes = init_bytes;
#endif
}
