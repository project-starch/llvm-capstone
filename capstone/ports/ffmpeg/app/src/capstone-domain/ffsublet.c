/* The Sublet port of FFmpeg's pools, in a Capstone domain: see ports/ffmpeg/sublet/ffsublet.h.
 *
 * A pool's storage is a list of blocks the Sublet heap lends LINEAR
 * (__capstone_sublet_malloc_linear, musl-capstone/runtime/sublet_heap.c). The heap keeps each
 * block's handle; the pool carves entries from the front of the uncarved remainder. So:
 *   new entry   sublet_carve: the entry's region, linear, in the entry's slot
 *   get         sublet_take: the slot keeps the handle, the caller gets an alias, narrowed
 *   return      sublet_give: one revoke, every alias of the entry dies, the region is linear again
 *   destroy     the heap's free of each block: one revoke, every entry carved from it dies
 * The records here, like FFmpeg's own BufferPoolEntry, live on the heap, outside every block. */
#include <stdio.h>
#include <stdlib.h>

#include "libavutil/ffsublet.h"
#include "sublet.h"

unsigned long __capstone_sublet_malloc_linear(size_t n, sublet_cap *out);
void __capstone_sublet_free_linear(unsigned long base);
void __capstone_sublet_heap_stats(unsigned long out[9]);

/* Entries per block the pool asks the heap for; the heap rounds up to its buddy order. */
#define FF_SUBLET_BLOCK_ENTRIES 4

struct FFSubletBlock {
    FFSubletBlock *next;
    unsigned long  base, cursor, end;
    sublet_cap     rest; /* the uncarved remainder, linear */
};

static struct {
    unsigned long blocks, destroyed, entries, takes, gives, destroy_revokes, destroy_merges;
} ff_sublet_counts;

void ff_sublet_pool_init(FFSubletPool *p, size_t size)
{
    p->blocks = NULL;
    p->entry  = (size + 15) & ~(size_t)15;
}

static FFSubletBlock *ff_sublet_block_new(FFSubletPool *p)
{
    FFSubletBlock *b = calloc(1, sizeof(*b));

    if (!b)
        return NULL;
    b->base = __capstone_sublet_malloc_linear(p->entry * FF_SUBLET_BLOCK_ENTRIES, &b->rest);
    if (!b->base) {
        free(b);
        return NULL;
    }
    b->cursor = b->base;
    b->end    = sublet_end(&b->rest);
    b->next   = p->blocks;
    p->blocks = b;
    ff_sublet_counts.blocks++;
    return b;
}

int ff_sublet_entry_new(FFSubletPool *p, FFSubletEntry *e)
{
    FFSubletBlock *b = p->blocks;

    if (!b || b->end - b->cursor < p->entry) {
        b = ff_sublet_block_new(p);
        if (!b)
            return -1;
    }
    b->cursor += p->entry;
    sublet_carve(&b->rest, b->cursor, (sublet_cap *)&e->slot);
    ff_sublet_counts.entries++;
    return 0;
}

void *ff_sublet_entry_take(FFSubletEntry *e, size_t size)
{
    sublet_cap   *slot = (sublet_cap *)&e->slot;
    unsigned long base = sublet_base(slot);
    char         *alias = sublet_take(slot);
    char         *p = alias + (base - __builtin_capstone_cap_get_cursor(alias));

    ff_sublet_counts.takes++;
    return __builtin_capstone_cap_shrink(p, base, base + size);
}

void ff_sublet_entry_give(FFSubletEntry *e)
{
    sublet_give((sublet_cap *)&e->slot);
    ff_sublet_counts.gives++;
}

void ff_sublet_pool_destroy(FFSubletPool *p)
{
    FFSubletBlock *b = p->blocks, *next;

    while (b) {
        next = b->next;
        /* The heap's revoke of its handle on the block: every entry carved from it dies. The
           revokes are MEASURED on the heap's own counter, and include its buddy merges. */
        unsigned long before[9], after[9];
        __capstone_sublet_heap_stats(before);
        __capstone_sublet_free_linear(b->base);
        __capstone_sublet_heap_stats(after);
        sublet_clear(&b->rest);
        ff_sublet_counts.destroyed++;
        ff_sublet_counts.destroy_revokes += after[7] - before[7];
        ff_sublet_counts.destroy_merges  += after[2] - before[2]; /* the heap's buddy merges */
        free(b);
        b = next;
    }
    p->blocks = NULL;
}

void ff_sublet_report(void);
void ff_sublet_report(void)
{
    printf("FFAPP-SUBLET-POOLS blocks=%lu destroyed=%lu entries=%lu takes=%lu gives=%lu "
           "destroy-revokes=%lu destroy-merges=%lu\n",
           ff_sublet_counts.blocks, ff_sublet_counts.destroyed, ff_sublet_counts.entries,
           ff_sublet_counts.takes, ff_sublet_counts.gives, ff_sublet_counts.destroy_revokes,
           ff_sublet_counts.destroy_merges);
}
