/* Capstone mode 2: fresh authority for each lease, revoked on pool return. */
FF2_INLINE void *ff2_sublet_issue(struct payload_block *b)
{
    return sublet_take(&b->region);
}
FF2_INLINE void ff2_sublet_return(struct payload_block *b)
{
    unsigned long before = sublet_stats.init;
    sublet_give(&b->region);
    if (sublet_stats.init != before) init_bytes += b->rounded;
    b->alias = NULL;
}
