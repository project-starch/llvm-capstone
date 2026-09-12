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
    /* These two carry the names aset.c gives them, so the manager's own
     * arithmetic over them needs no patch: it adds, subtracts and compares
     * them and never dereferences one. They are integers and not pointers
     * because the manager has no capability to the inside of a block, and a
     * pointer here would be one. Declaring them as integers is also how the
     * patch was found: every place aset.c casts a cursor to a chunk fails to
     * compile, and those are exactly the places that have to carve instead. */
    unsigned long freeptr;          /* the next chunk starts here */
    unsigned long endptr;           /* one past the block's last byte */
    unsigned long base;             /* where its chunks begin */
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
    /* Which block the chunk was carved from, so that a chunk which sat on a
     * size-class free list can still say it when it goes out again. The
     * manager wrote that into the chunk's own header before; a freed chunk is
     * revoked here, so it cannot hold it while it waits. */
    unsigned int block;
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

/* The manager says when a block it just took is a keeper it is carving again
 * after a reset. The level below cannot tell: a keeper is a block like any
 * other to it, and that is the point. */
void pg_subpool_count_keeper(void);

/* A chunk's header cannot say which block it belongs to the way the manager
 * says it, by the offset from the chunk back to the block: a block's header is
 * no longer at the front of the block, and a capability cannot be made to
 * reach out of the chunk it was given for anyway. So the header carries the
 * two indices instead, and these turn them back into the tables' entries. An
 * index is four bytes and a capability is sixteen, which is the whole reason
 * the indirection is cheaper than it looks. */
pg_block *pg_subpool_block_at(unsigned int i);
unsigned int pg_subpool_block_index(pg_block *b);

/* A block's chunks, in the order they were carved. This is the only way to
 * walk them: the manager used to do it by casting the block's cursor to a
 * chunk and stepping by each chunk's size, and it has no capability to the
 * inside of a block to do that with. It also cannot read a free chunk at all,
 * so what a walk can check is the bookkeeping and not the bytes. */
unsigned int pg_subpool_first_chunk(pg_block *b);
unsigned int pg_subpool_next_chunk(unsigned int i);

/* A context's header, and it has to come from here rather than from a
 * sub-pool. aset.c puts AllocSetContext and the first block in one malloc and
 * keeps the block across a reset because of it. One revocation cannot spare
 * part of what it covers, so a header inside the sub-pool would be destroyed
 * by the very reset it is executing. See sublet/README.md, "The keeper block".
 *
 * The headers are a table of equal slots in the domain's own data, because
 * they are few, they are all about the same size, and a table cannot be
 * revoked. PG_HEADER_BYTES is checked against the manager's largest context
 * struct by a static assertion in the patch, so a version that grew one would
 * fail to build rather than overrun. */
/* sizeof(AllocSetContext) is about 290 bytes with sixteen-byte pointers, and a
 * static assertion in the patch fails the build if a version of the manager
 * grows past this. */
#define PG_HEADER_BYTES 320u
void *pg_subpool_header(unsigned long bytes);
void pg_subpool_header_free(void *p);

/* The counts a pass reports. Revocations are the claim; the rest says what it
 * cost to get there. */
struct pg_subpool_counts {
    unsigned long created, destroyed, resets, resets_empty;
    unsigned long revocations, handles;
    unsigned long blocks, blocks_freed, grown;
    unsigned long blocks_live, blocks_peak;
    /* Blocks carved again as a keeper after a reset, which upstream does not
     * do at all: it keeps the keeper and this port revokes it with the rest
     * and carves it again, because a revocation cannot spare part of what it
     * covers. Counted apart so that "blocks taken" can be compared with the
     * unprotected arm's, which is the comparison the paper makes. */
    unsigned long keepers;
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

/* The tables, sized from the recording at about four times its peak, and the
 * factor is four rather than ten for a reason that cost a run.
 *
 * They are static, in the domain image, because a table that grew would need
 * an allocator under it and that allocator would need a table. So they are the
 * image, and the image has a ceiling: the module asks the buddy allocator for
 * the image doubled, rounded up to a power-of-two page count, and that
 * allocator stops at order ten, four megabytes. An image much above two
 * megabytes therefore cannot be created at all, and it fails as
 * `create_dom failed`, which says nothing about tables. The first Sublet
 * image was 2.5 MB of which two were these tables, and that is how the
 * ceiling was found.
 *
 * The peaks the recording gives, on the tpcb rung: 120 contexts alive, 241
 * blocks held, 2 321 chunk entries held. Four times each, and the four tables
 * together are about 640 KiB.
 *
 * A table that runs out is a fault at a named place and not a corruption:
 * every allocator here returns null or zero, and the manager's own
 * out-of-memory path takes it from there. */
#ifndef PG_SUBPOOL_MAX
#define PG_SUBPOOL_MAX    512u
#endif
#ifndef PG_BLOCK_MAX
#define PG_BLOCK_MAX     1024u
#endif
#ifndef PG_CHUNK_MAX
#define PG_CHUNK_MAX     8192u
#endif

#endif
