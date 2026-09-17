/* Native functional control: ordinary pointers, without capability revocation. */
FF2_INLINE void payload_init_region(void *p, size_t n)
{
    (void)n;
    remaining.c = p; payload_base = (uintptr_t)p;
}
static void *spatial_alias(struct payload_block *b) { return b->region.c; }
FF2_INLINE void *payload_issue_pointer(struct payload_block *b) { return b->full_alias; }
static int same_authority(const void *a, const void *b) { return a == b; }
FF2_INLINE void payload_carve(struct payload_block *b, size_t rounded)
{
    (void)rounded;
    b->region.c = (unsigned char *)remaining.c + payload_used;
}
FF2_INLINE void payload_prepare_backing(struct payload_block *b)
{
    if (mode != 2 || !b->full_alias) b->full_alias = spatial_alias(b);
}
FF2_INLINE void payload_return_lease(struct payload_block *b) { (void)b; }
FF2_INLINE void payload_free_backing(struct payload_block *b) { (void)b; }
FF2_INLINE void payload_report_stats(struct ff2_header *h) { (void)h; }
