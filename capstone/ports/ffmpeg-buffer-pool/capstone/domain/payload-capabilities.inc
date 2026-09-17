/* Capstone bounds/backing protection; mode 2 adds Sublet pool leases. */
#include "../../sublet/pool-leases.inc"

FF2_INLINE void payload_init_region(void *p, size_t n)
{
    sublet_store(&remaining, p);
    payload_base = sublet_base(&remaining);
    if (sublet_type(&remaining) != 0 || sublet_end(&remaining) - payload_base != n) ff2_fail(302);
}
static void *spatial_alias(struct payload_block *b)
{
    void *p;
    __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(%1)\n"
                     ".insn s 0x5b, 0x4, x0, 0(%1)\n"
                     ".insn r 0x5b, 0x1, 0x03, %0, x0, x0\n"
                     : "=&r"(p) : "r"(&b->region) : "memory");
    sublet_stats.delin++;
    return p;
}
FF2_INLINE void *payload_issue_pointer(struct payload_block *b)
{
    void *p = mode == 2 ? ff2_sublet_issue(b) : b->full_alias;
    return __builtin_capstone_cap_shrink(p, b->address, b->address + b->requested);
}
/* Address equality cannot authorize free. Compare the tagged current lease;
 * LCC's validity selector is unimplemented in the pinned QEMU. */
static int same_authority(const void *a, const void *b)
{
    sublet_cap aa, bb;
    sublet_store(&aa, (void *)a); sublet_store(&bb, (void *)b);
    if (sublet_type(&aa) != 1 || sublet_type(&bb) != 1) return 0;
    const volatile uint64_t *x = (const volatile uint64_t *)&aa;
    const volatile uint64_t *y = (const volatile uint64_t *)&bb;
    return x[0] == y[0] && x[1] == y[1];
}
FF2_INLINE void payload_carve(struct payload_block *b, size_t rounded)
{
    sublet_carve(&remaining, b->address + rounded, &b->region);
}
FF2_INLINE void payload_prepare_backing(struct payload_block *b)
{
    if (mode) sublet_handle(&b->region, &b->outer);
    if (mode != 2 || !b->full_alias) {
        if (mode != 2 && !b->full_alias) b->full_alias = spatial_alias(b);
    }
}
FF2_INLINE void payload_return_lease(struct payload_block *b)
{
    if (mode == 2) ff2_sublet_return(b);
}
FF2_INLINE void payload_free_backing(struct payload_block *b)
{
    if (mode) {
        unsigned long before = sublet_stats.init;
        sublet_give_to(&b->outer, &b->region);
        if (sublet_stats.init != before) init_bytes += b->rounded;
        b->full_alias = b->alias = NULL;
    }
}
FF2_INLINE void payload_report_stats(struct ff2_header *h)
{
    h->split = sublet_stats.split; h->mrev = sublet_stats.mrev;
    h->delin = sublet_stats.delin; h->revoke = sublet_stats.revoke;
    h->init = sublet_stats.init; h->init_bytes = init_bytes;
}
