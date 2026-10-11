/* The Sublet port of FFmpeg's pools: what they need from the level below.
 *
 * Copied into libavutil/ by the app port's `prepare-source.sh --sublet`, and included by the
 * patched pools under FF_SUBLET_POOLS only. The implementation is the app's, in
 * src/capstone-domain/ffsublet.c, over the Sublet heap's linear lend.
 *
 * A pool takes its storage from the heap LINEAR, the way SQLite's lookaside takes its block from
 * memsys5: the heap keeps the block's handle, the pool carves entries from it front to back. A get
 * takes the entry's handle and hands out an alias; a return gives it back, one revoke; and the
 * pool's destruction is the heap's free of each block, one revoke per block, which ends every entry
 * carved from it. */
#ifndef AVUTIL_FFSUBLET_H
#define AVUTIL_FFSUBLET_H

#include <stddef.h>

/* One owned capability. It is moved by the implementation and never copied. */
typedef struct FFSubletSlot {
    void *c;
} FFSubletSlot;

typedef struct FFSubletBlock FFSubletBlock;

/* A pool's storage. entry == 0 means the pool is not ported and keeps upstream's path. */
typedef struct FFSubletPool {
    FFSubletBlock *blocks; /* the block entries are carved from first */
    size_t         entry;  /* the bytes each entry's region spans, a multiple of 16 */
} FFSubletPool;

/* An entry's region while it is free, its handle while it is out. */
typedef struct FFSubletEntry {
    FFSubletSlot slot;
} FFSubletEntry;

void  ff_sublet_pool_init(FFSubletPool *p, size_t size);
/* Carve a new entry, taking a block from the heap when the current one is used up. 0 on success. */
int   ff_sublet_entry_new(FFSubletPool *p, FFSubletEntry *e);
/* Hand the free entry out: an alias of exactly `size` bytes. */
void *ff_sublet_entry_take(FFSubletEntry *e, size_t size);
/* The entry comes back: every alias of it dies. */
void  ff_sublet_entry_give(FFSubletEntry *e);
/* The pool's end, for an entry already given back: it is NOT given back again -- its block's
 * revoke ends it with every other entry. With want_alias, an alias of `size` bytes through a new
 * handle, for a callback that must still read the entry; the handle is dropped at once, so the
 * alias lives exactly until the block's revoke. Returns the alias, or NULL. */
void *ff_sublet_entry_end(FFSubletEntry *e, size_t size, int want_alias);
/* One revoke per block, every entry with it; the blocks go back to the heap. */
void  ff_sublet_pool_destroy(FFSubletPool *p);
/* This file's own counts, for a fixture that measures across FFmpeg's code: [0] the revokes this
 * translation unit performed (sublet.h counts per file), [1] gives, [2] ends. */
void  ff_sublet_counts(unsigned long out[3]);

/* An address, read from a capability's cursor. Reading it never faults, even on a revoked
 * capability, and it is only ever compared, never turned back into a pointer. */
static inline unsigned long ff_sublet_addr(const void *p)
{
    return __builtin_capstone_cap_get_cursor((void *)p);
}

#endif /* AVUTIL_FFSUBLET_H */
