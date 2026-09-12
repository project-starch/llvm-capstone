/* The level below PostgreSQL's memory manager, under Sublet: one sub-pool per
 * context, one handle senior to it, and a block table beside it.
 *
 * Why a sub-pool and not a block: aset.c takes its blocks one at a time, and a
 * handle from mrev is senior to one node, so a handle kept per block would
 * make a reset cost one revocation per block and the cost of a reset would
 * grow with the objects in the context. The claim C11 makes is one revocation
 * per context, so the delegation has to be per context. sublet/README.md
 * argues it and prices the size.
 *
 * Three things live outside the sub-pool, in memory no revocation reaches, and
 * each for the same reason: a revocation cannot spare part of what it covers.
 *
 *   the sub-pool table   its own capability and its handle. A revoke that
 *                        reached these would destroy the means to recover.
 *   the block table      what aset.c calls AllocBlockData. The manager reads
 *                        and writes a block's header across a reset of the
 *                        block's own contents, and it walks the block list
 *                        while tearing the context down.
 *   the context header   AllocSetContext, which the manager is executing on
 *                        when it resets. See sublet/README.md.
 *
 * So a block's header is not at the front of the block any more. That is the
 * one structural change, and it is why aset.c's own field accesses need no
 * patch: AllocBlock was always a pointer, and it still is, to an entry here.
 */
#ifndef PG_SUBPOOL_H
#define PG_SUBPOOL_H

#include "sublet.h"

/* A block, as the manager sees it, plus what the discipline needs.
 *
 * `region` holds the block's bytes, linear, while the block is being carved
 * into chunks. It is never copied into a C variable: every use goes through a
 * primitive that takes its address.
 *
 * `base` and `limit` are addresses and not pointers, on purpose. The manager
 * bumps a cursor through a block, and a cursor is arithmetic; the capability
 * for each chunk comes from the carve, not from the cursor. Keeping them as
 * integers makes it impossible to dereference one by accident.
 */
typedef struct pg_block {
    sublet_cap region;              /* the uncarved tail of the block */
    unsigned long base;             /* where its chunks begin */
    unsigned long limit;            /* one past its last byte */
    unsigned long cursor;           /* the next chunk starts here */
    struct pg_block *prev;          /* the manager's block list */
    struct pg_block *next;
    void *aset;                     /* the context it belongs to */
    struct pg_subpool *pool;        /* the sub-pool it was carved from */
    struct pg_block *free_next;     /* the table's free list, when unused */
    unsigned int chunk_head;        /* its chunk entries, to return at once */
    unsigned int chunk_tail;
    unsigned int chunks;            /* how many, so a release can count down */
    /* The level below's own chain of the blocks it carved from one sub-pool,
     * kept apart from prev and next above, which are the manager's. Without it
     * a reset would have to trust the manager to hand every block back before
     * it asks for the revocation, and a level that trusts the level above to
     * free correctly is not a level that can claim anything. */
    struct pg_block *pool_prev, *pool_next;
} pg_block;

/* A chunk, as the discipline needs it. aset.c keeps a freed chunk's free-list
 * link inside the chunk, and a revoked chunk cannot hold it, so the link moves
 * here. So does the size class, because the manager reads a freed chunk's class
 * when it pops it, and so does the capability itself.
 *
 * `slot` holds the chunk's region while the chunk is free and its handle while
 * the chunk is out, which is what sublet_take and sublet_give do to one slot.
 * An entry is therefore held from the carve until the reset that reclaims its
 * block, and not released by a free: the recording says 2 321 are held at the
 * peak of the tpcb rung, against 2 293 chunks alive.
 *
 * `block_next` chains an entry to the others carved from the same block, so a
 * block returns all of its entries at once. It is never walked to find
 * anything, only spliced, which is why a reset does not touch a chunk.
 */
typedef struct pg_chunk {
    sublet_cap slot;
    unsigned int next_free;         /* the manager's size-class list, by index */
    /* Two lists share this field, and they never overlap: while the entry
     * belongs to a block it is that block's chain, and while the entry is
     * unused it is the level below's list of spare entries. An entry that
     * belongs to no block is exactly an entry that is spare, so one field
     * suffices and a block's entries go back in one splice rather than a walk. */
    unsigned int block_next;
    unsigned int bytes;             /* what the class gives, not what was asked */
    unsigned int _pad;
} pg_chunk;

typedef struct pg_subpool pg_subpool;

/* The arena the host gave the domain, once. Sub-pools are carved from it and
 * never joined again, so they are all one size and a returned one goes on a
 * free list. That is the second reason the size is fixed rather than a ladder:
 * a region can be split and not joined, so equal sizes make reuse exact. */
/* Returns the type of the capability it was handed, so a caller can say what
 * went wrong instead of tripping an assertion inside the first primitive.
 * Zero is linear, which is the only type this level can carve and revoke; one
 * is non-linear, two revocable, three uninitialised, seven an empty slot. A
 * region that is not linear cannot be delegated, because a handle senior to it
 * would have nothing to revoke. */
unsigned long pg_subpool_arena(void *arena, unsigned long bytes);

/* 65536, from the recording: 16.4 MiB of arena at the peak of the tpcb rung
 * and 99.96% of the resets and deletes at exactly one revocation. */
#define PG_SUBPOOL_BYTES 65536u

pg_subpool *pg_subpool_create(void);
void pg_subpool_destroy(pg_subpool *sp);

/* One revocation, and none at all when nothing was carved since the last one.
 * A third of the resets in the recording are that case. */
void pg_subpool_reset(pg_subpool *sp);

/* A block of at least `bytes`, or null when the sub-pool has no room left.
 * The caller asks again after pg_subpool_grow. */
pg_block *pg_subpool_block(pg_subpool *sp, unsigned long bytes);

/* Another sub-pool for a context that filled the first: it costs one more
 * revocation at the next reset, which the recording says happens eight times
 * in thirty thousand. Returns zero when the arena is exhausted. */
int pg_subpool_grow(pg_subpool *sp, unsigned long bytes);

/* Give one block back before a reset. aset.c does this for a block it empties
 * and for an external chunk it frees. */
void pg_subpool_block_free(pg_block *b);

/* Carve the next `bytes` out of `b`, and give back the index of the entry that
 * holds them, or zero when the block has no room. Index zero is never an
 * entry, so it doubles as "none" the way a null pointer would.
 *
 * The carve does not hand the chunk out. pg_subpool_hand does that, and the
 * two are separate because the manager carves a chunk once and hands it out
 * again after every free. */
unsigned int pg_subpool_carve(pg_block *b, unsigned long bytes);

/* The chunk goes to the program: the entry keeps the handle, the program gets
 * an alias, and pg_subpool_drop kills the alias again. */
void *pg_subpool_hand(unsigned int i);
void pg_subpool_drop(unsigned int i);

/* The entry behind an index, for the manager's free lists and size classes. */
pg_chunk *pg_subpool_entry(unsigned int i);

/* The counts a pass reports. Revocations are the claim; the rest says what it
 * cost to get there. */
struct pg_subpool_counts {
    unsigned long created, destroyed, resets, resets_empty;
    unsigned long revocations, handles;
    unsigned long blocks, blocks_freed, grown;
    unsigned long carves, hands, drops;
    unsigned long entries_live, entries_peak;
    unsigned long pools_live, pools_peak;
};
extern struct pg_subpool_counts pg_subpool_counts;

/* What the primitives themselves ran, which is a different question from what
 * this level was asked to do and the one the board answers. sublet.h keeps its
 * tally per translation unit, and the primitives run in this one, so a reader
 * elsewhere has to come through here.
 *
 * The number that matters on silicon is split plus mrev. Each allocates a
 * revocation node from a bump head with no reclamation, so a run spends nodes
 * for as long as it runs and a revoke does not give any back. The resident
 * bitstream carries 65 536 of them. */
const struct sublet_stats *pg_subpool_primitives(void);

/* The tables, sized from the recording with room above it. 120 contexts are
 * alive at the peak of the tpcb rung and 241 blocks and 2 321 chunk entries
 * are held, so each table is an order above what was seen and the whole of it
 * is under a megabyte. A table that runs out is a fault at a named place and
 * not a corruption: every allocator here returns null instead. */
#define PG_SUBPOOL_MAX   1024u      /* a 64 MiB arena holds this many */
#define PG_BLOCK_MAX     4096u
#define PG_CHUNK_MAX    32768u      /* one megabyte at 32 bytes an entry */

#endif
