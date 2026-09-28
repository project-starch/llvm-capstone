/* The Sublet port of FFmpeg's pools, in a Capstone domain: see ports/ffmpeg/sublet/ffsublet.h.
 *
 * A pool's storage is a list of blocks the Sublet heap lends LINEAR
 * (__capstone_sublet_malloc_linear, musl-capstone/runtime/sublet_heap.c). The heap keeps each
 * block's handle; the pool carves entries from the front of the uncarved remainder. So:
 *   new entry   sublet_carve: the entry's region, linear, in the entry's slot
 *   get         sublet_take: the slot keeps the handle, the caller gets an alias, narrowed
 *   return      sublet_give: one revoke, every alias of the entry dies, the region is linear again
 *   end         the pool is ending: no give -- the entry's slot is dropped, and the block's revoke
 *               below ends it (a callback that must read it gets an alias through a new handle)
 *   destroy     the heap's free of each block: one revoke, every entry carved from it dies, then
 *               the block's record, one more heap free
 * The pool takes no senior handle of its own (the design note in docs/plans considered one): the
 * heap keeps its handle on every block it lends, and its free revokes that handle, which already
 * covers everything carved from the block.
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
    unsigned long blocks, destroyed, entries, takes, gives, ends, end_takes, destroy_merges;
} counts;

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
    counts.blocks++;
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
    counts.entries++;
    return 0;
}

void *ff_sublet_entry_take(FFSubletEntry *e, size_t size)
{
    sublet_cap   *slot = (sublet_cap *)&e->slot;
    unsigned long base = sublet_base(slot);
    char         *alias = sublet_take(slot);
    char         *p = alias + (base - __builtin_capstone_cap_get_cursor(alias));

    counts.takes++;
    return __builtin_capstone_cap_shrink(p, base, base + size);
}

void ff_sublet_entry_give(FFSubletEntry *e)
{
    sublet_give((sublet_cap *)&e->slot);
    counts.gives++;
}

void *ff_sublet_entry_end(FFSubletEntry *e, size_t size, int want_alias)
{
    void *p = NULL;

    if (want_alias) {
        p = ff_sublet_entry_take(e, size);
        counts.end_takes++;
    }
    sublet_clear((sublet_cap *)&e->slot);
    counts.ends++;
    return p;
}

void ff_sublet_pool_destroy(FFSubletPool *p)
{
    FFSubletBlock *b = p->blocks, *next;

    while (b) {
        next = b->next;
        /* The heap's revoke of its handle on the block: every entry carved from it dies. By
           __capstone_sublet_free_linear's own code that is one revoke plus one per buddy merge,
           so only the merges are counted here; a bracket of the revoke count around that one call
           would measure its code, not the pool (audit, 2026-09-29). */
        unsigned long before[9], after[9];
        __capstone_sublet_heap_stats(before);
        __capstone_sublet_free_linear(b->base);
        __capstone_sublet_heap_stats(after);
        sublet_clear(&b->rest);
        counts.destroyed++;
        counts.destroy_merges += after[2] - before[2]; /* the heap's buddy merges */
        free(b);
        b = next;
    }
    p->blocks = NULL;
}

void ff_sublet_counts(unsigned long out[3])
{
    out[0] = sublet_stats.revoke; /* THIS file's: sublet.h keeps one copy per translation unit */
    out[1] = counts.gives;
    out[2] = counts.ends;
}

void ff_sublet_report(void);
void ff_sublet_report(void)
{
    printf("FFAPP-SUBLET-POOLS blocks=%lu destroyed=%lu entries=%lu takes=%lu gives=%lu ends=%lu "
           "end-takes=%lu file-revokes=%lu destroy-merges=%lu\n",
           counts.blocks, counts.destroyed, counts.entries, counts.takes, counts.gives,
           counts.ends, counts.end_takes, sublet_stats.revoke, counts.destroy_merges);
}
