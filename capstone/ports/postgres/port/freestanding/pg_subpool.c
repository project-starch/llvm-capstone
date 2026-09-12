/* The level below PostgreSQL's memory manager, under Sublet.
 *
 * pg_level0.c beside this file is the same level for the unprotected arm: a
 * first-fit allocator over one region, which is what the manager talks to when
 * nothing is being enforced. This file replaces it for the protected arm, and
 * the two are never linked together.
 *
 * What it does differently is one thing. The unprotected level hands the
 * manager bytes. This one hands the manager bytes and keeps a capability
 * senior to them, one per context, so that the context's whole heap can be
 * taken back in a single instruction. Everything else here follows from having
 * to keep that capability somewhere a revocation cannot reach.
 *
 * The reasoning is in ../../sublet/README.md, and the sizes below are measured
 * rather than chosen. Nothing in this file allocates: the three tables are
 * static, because a table that grew would need an allocator under it and that
 * allocator would need a table.
 */
#include <stddef.h>
#include <stdint.h>
#include "pg_subpool.h"

struct pg_subpool_counts pg_subpool_counts;

/* A context's sub-pool. `pool` holds what is not carved yet; `handle` is
 * senior to the whole of it, so one revoke takes back every block and every
 * chunk and every alias the program still holds into any of them.
 *
 * `extra` is the second sub-pool a context gets when it fills the first. It
 * costs one more revocation at the next reset, which the recording says
 * happens eight times in thirty thousand, so it is a list and not a table. */
struct pg_subpool {
    sublet_cap pool;
    sublet_cap handle;
    unsigned long cursor;           /* where the next block starts */
    unsigned long limit;
    unsigned int carved;            /* blocks taken since the last revoke */
    pg_block *blocks;               /* what it carved, the level below's chain */
    struct pg_subpool *extra;
    struct pg_subpool *free_next;
};

/* The context headers, in the domain's own data. Sixteen-byte aligned because
 * a header holds capabilities, and a capability stored to a slot that is not
 * sixteen-aligned loses its tag. */
static unsigned char header_tab[PG_SUBPOOL_MAX][PG_HEADER_BYTES]
    __attribute__((aligned(16)));
static unsigned int header_free[PG_SUBPOOL_MAX];
static unsigned int header_free_n;
static unsigned int header_next;

static struct pg_subpool pool_tab[PG_SUBPOOL_MAX];
static pg_block block_tab[PG_BLOCK_MAX];
static pg_chunk chunk_tab[PG_CHUNK_MAX];

static struct pg_subpool *pool_free;    /* sub-pools handed back, whole */
static pg_block *block_free;
/* Spare entries, by index because zero means none. The link is block_next:
 * a spare entry belongs to no block, so the two uses of that field cannot
 * meet. Using next_free here instead was the second run's bug, and it showed
 * as a free list one entry long, because a block's chain is in block_next and
 * the splice walked the other field. */
static unsigned int chunk_free;

/* The arena, and the part of it not yet cut into sub-pools. A sub-pool is
 * never joined back, so the cut is one way and a returned sub-pool goes on
 * pool_free instead. That is why the size is fixed: equal sizes make reuse
 * exact, where a ladder would leave holes nothing fits. */
static sublet_cap arena;
static unsigned long arena_cursor, arena_limit;
static int arena_ready;

#define ALIGN16(n) (((n) + 15ul) & ~15ul)

unsigned long
pg_subpool_arena(void *region, unsigned long bytes)
{
    unsigned long ty;

    /* The region arrives in a register from the host and goes straight into
     * its slot. It must not pass through a C variable of pointer type: a
     * linear capability stored anywhere but a slot is a capability the
     * compiler may copy, and a copy of a linear capability is what the
     * hardware refuses. */
    sublet_store(&arena, region);
    ty = sublet_type(&arena);
    if (ty != 0)                    /* not linear: nothing here can work */
        return ty;
    arena_cursor = sublet_base(&arena);
    arena_limit = arena_cursor + bytes;
    if (arena_limit > sublet_end(&arena))
        arena_limit = sublet_end(&arena);

    for (unsigned int i = 0; i + 1 < PG_BLOCK_MAX; i++)
        block_tab[i].free_next = &block_tab[i + 1];
    block_free = &block_tab[0];

    /* Entry zero is never handed out, so an index of zero means none and no
     * caller needs a second field to say so. */
    for (unsigned int i = 1; i + 1 < PG_CHUNK_MAX; i++)
        chunk_tab[i].block_next = i + 1;
    chunk_tab[PG_CHUNK_MAX - 1].block_next = 0;
    chunk_free = 1;

    for (unsigned int i = 0; i + 1 < PG_SUBPOOL_MAX; i++)
        pool_tab[i].free_next = &pool_tab[i + 1];
    pool_free = &pool_tab[0];

    arena_ready = 1;
    return 0;
}

/* One sub-pool's worth of the arena into `slot`, and a handle senior to it
 * into `handle`. The handle is taken after the carve and not before, because
 * sublet_handle makes a node senior to what the slot holds now. */
static int
cut(sublet_cap *slot, sublet_cap *handle, unsigned long bytes)
{
    if (!arena_ready || arena_cursor + bytes > arena_limit)
        return 0;
    arena_cursor += bytes;
    sublet_carve(&arena, arena_cursor, slot);
    sublet_handle(slot, handle);
    pg_subpool_counts.handles++;
    return 1;
}

static struct pg_subpool *
pool_alloc(unsigned long bytes)
{
    struct pg_subpool *sp = pool_free;

    if (!sp)
        return NULL;
    if (bytes < PG_SUBPOOL_BYTES)
        bytes = PG_SUBPOOL_BYTES;
    bytes = ALIGN16(bytes);

    /* An entry on the free list may still hold the region it was given, and
     * then it is reused rather than cut again. This is the whole reason the
     * size is fixed: a region splits and does not join, so arena bytes handed
     * out are gone for good unless the next context can take them exactly as
     * they are. Without this the arena empties after one sub-pool per
     * 64 KiB of it, which is how the first run of the test ended: null from
     * the five hundred and thirteenth create. */
    if (sublet_type(&sp->pool) == 7) {
        if (!cut(&sp->pool, &sp->handle, bytes))
            return NULL;
    } else if (sublet_end(&sp->pool) - sublet_base(&sp->pool) < bytes) {
        return NULL;                /* a reused sub-pool is too small for this */
    }
    pool_free = sp->free_next;
    sp->free_next = NULL;
    sp->extra = NULL;
    sp->blocks = NULL;
    sp->cursor = sublet_base(&sp->pool);
    sp->limit = sublet_end(&sp->pool);
    sp->carved = 0;
    if (++pg_subpool_counts.pools_live > pg_subpool_counts.pools_peak)
        pg_subpool_counts.pools_peak = pg_subpool_counts.pools_live;
    return sp;
}

pg_subpool *
pg_subpool_create(void)
{
    struct pg_subpool *sp = pool_alloc(PG_SUBPOOL_BYTES);

    if (sp)
        pg_subpool_counts.created++;
    return sp;
}

int
pg_subpool_grow(pg_subpool *sp, unsigned long bytes)
{
    struct pg_subpool *more = pool_alloc(bytes);

    if (!more)
        return 0;
    /* At the front of the chain, so the next block comes from the newest
     * sub-pool and the older ones are not scanned again. */
    more->extra = sp->extra;
    sp->extra = more;
    pg_subpool_counts.grown++;
    return 1;
}

/* Every chunk entry of `b` back to the free list, in one splice, and the block
 * itself back to the block table. The entries are not touched: their slots
 * hold capabilities that the revoke has already killed, and the next carve
 * stores over them. That is what keeps a reset off the chunks. */
static void
block_release(pg_block *b)
{
    if (b->chunk_head) {
        chunk_tab[b->chunk_tail].block_next = chunk_free;
        chunk_free = b->chunk_head;
        b->chunk_head = b->chunk_tail = 0;
        pg_subpool_counts.entries_live -= b->chunks;
        b->chunks = 0;
    }
    /* Out of its sub-pool's chain. */
    if (b->pool_prev)
        b->pool_prev->pool_next = b->pool_next;
    else if (b->pool)
        b->pool->blocks = b->pool_next;
    if (b->pool_next)
        b->pool_next->pool_prev = b->pool_prev;
    b->pool_prev = b->pool_next = NULL;
    b->prev = b->next = NULL;
    b->aset = NULL;
    b->pool = NULL;
    b->free_next = block_free;
    block_free = b;
}

/* What a revoke leaves behind, for one sub-pool of a chain. */
/* Every block the sub-pool carved, back to the block table, and with them
 * their chunk entries. This is a walk, and it is the same walk aset.c does
 * over its own block list on a reset, so it adds no order of work. What it
 * does not do is touch a chunk: the entries go back in one splice per block. */
static void
pool_drop_blocks(struct pg_subpool *sp)
{
    pg_block *b = sp->blocks;

    while (b) {
        pg_block *next = b->pool_next;

        sublet_clear(&b->region);
        b->pool_prev = b->pool_next = NULL;
        b->pool = NULL;             /* so block_release does not walk back in */
        block_release(b);
        pg_subpool_counts.blocks_freed++;
        pg_subpool_counts.blocks_live--;
        b = next;
    }
    sp->blocks = NULL;
}

static void
pool_revoke(struct pg_subpool *sp)
{
    pool_drop_blocks(sp);
    sublet_give_to(&sp->handle, &sp->pool);
    pg_subpool_counts.revocations++;
    sublet_handle(&sp->pool, &sp->handle);
    pg_subpool_counts.handles++;
    sp->cursor = sublet_base(&sp->pool);
    sp->carved = 0;
}

void
pg_subpool_reset(pg_subpool *sp)
{
    pg_subpool_counts.resets++;

    /* A sub-pool nothing was carved from holds nothing, so there is nothing to
     * revoke. A third of the resets in the recording are this case, almost all
     * of them ExprContext, which is created and deleted without ever holding
     * an object. */
    int any = sp->carved != 0;

    for (struct pg_subpool *e = sp->extra; e; e = e->extra)
        any |= e->carved != 0;
    if (!any) {
        pg_subpool_counts.resets_empty++;
        return;
    }

    if (sp->carved)
        pool_revoke(sp);
    for (struct pg_subpool *e = sp->extra; e; e = e->extra)
        if (e->carved)
            pool_revoke(e);
}

static void
pool_return(struct pg_subpool *sp)
{
    /* The sub-pool is not cut back into the arena, because a region splits and
     * does not join. It goes on the free list WITH its region still in its
     * slot, and the next create takes it as it is. That is exact because every
     * sub-pool is the same size, and it is the second reason the size is fixed
     * rather than a ladder. */
    pool_drop_blocks(sp);           /* nothing was revoked if carved was zero */
    sp->extra = NULL;
    sp->free_next = pool_free;
    pool_free = sp;
    pg_subpool_counts.pools_live--;
}

void
pg_subpool_destroy(pg_subpool *sp)
{
    struct pg_subpool *e = sp->extra;

    if (sp->carved)
        pool_revoke(sp);
    while (e) {
        struct pg_subpool *next = e->extra;

        if (e->carved)
            pool_revoke(e);
        pool_return(e);
        e = next;
    }
    pool_return(sp);
    pg_subpool_counts.destroyed++;
}

/* A block out of the first sub-pool in the chain with room for it. */
static pg_block *
block_from(struct pg_subpool *sp, unsigned long bytes)
{
    pg_block *b;

    if (sp->cursor + bytes > sp->limit)
        return NULL;
    b = block_free;
    if (!b)
        return NULL;
    block_free = b->free_next;
    b->free_next = NULL;

    sp->cursor += bytes;
    sublet_carve(&sp->pool, sp->cursor, &b->region);
    sp->carved++;

    b->base = sublet_base(&b->region);
    b->endptr = sublet_end(&b->region);
    b->freeptr = b->base;
    b->prev = b->next = NULL;
    b->aset = NULL;
    b->pool = sp;
    b->chunk_head = b->chunk_tail = 0;
    b->chunks = 0;
    b->pool_prev = NULL;
    b->pool_next = sp->blocks;
    if (sp->blocks)
        sp->blocks->pool_prev = b;
    sp->blocks = b;
    pg_subpool_counts.blocks++;
    if (++pg_subpool_counts.blocks_live > pg_subpool_counts.blocks_peak)
        pg_subpool_counts.blocks_peak = pg_subpool_counts.blocks_live;
    return b;
}

pg_block *
pg_subpool_block(pg_subpool *sp, unsigned long bytes)
{
    bytes = ALIGN16(bytes);

    pg_block *b = block_from(sp, bytes);

    if (b)
        return b;
    for (struct pg_subpool *e = sp->extra; e; e = e->extra) {
        b = block_from(e, bytes);
        if (b)
            return b;
    }
    return NULL;
}

void
pg_subpool_block_free(pg_block *b)
{
    /* One block back before a reset. Its region cannot go back to the
     * sub-pool, because a region splits and does not join, so the bytes stay
     * where they are until the reset revokes the sub-pool. What is returned is
     * the bookkeeping: the block entry and its chunk entries.
     *
     * This is the one place the port is weaker than the manager it carries. On
     * the host, free() gives the bytes back and another context can have them.
     * Here they are the context's until it resets. The recording says what
     * that costs: 16.4 MiB of arena at the peak against 64 MiB available,
     * because the manager frees whole blocks only when a context empties one
     * and it resets far more often than that.
     *
     * The block's own capability is dropped rather than revoked. Every alias
     * to a chunk in it stays alive until the sub-pool is revoked, so this is
     * not a free in the security sense and the port does not claim it is.
     * aset.c calls it for a block it has emptied, which means no chunk in it
     * is reachable from the program any more. */
    sublet_clear(&b->region);
    block_release(b);
    pg_subpool_counts.blocks_freed++;
    pg_subpool_counts.blocks_live--;
}

void
pg_subpool_count_keeper(void)
{
    pg_subpool_counts.keepers++;
}

unsigned int
pg_subpool_carve(pg_block *b, unsigned long bytes)
{
    unsigned int i = chunk_free;
    unsigned long end;

    bytes = ALIGN16(bytes);
    end = b->freeptr + bytes;
    if (end > b->endptr || !i)
        return 0;

    chunk_free = chunk_tab[i].block_next;
    b->freeptr = end;
    sublet_carve(&b->region, end, &chunk_tab[i].slot);
    chunk_tab[i].next_free = 0;
    chunk_tab[i].bytes = (unsigned int) bytes;
    chunk_tab[i].block = (unsigned int) (b - block_tab);

    /* Onto the tail of the block's chain, so a release is one splice. */
    chunk_tab[i].block_next = 0;
    if (b->chunk_tail)
        chunk_tab[b->chunk_tail].block_next = i;
    else
        b->chunk_head = i;
    b->chunk_tail = i;
    b->chunks++;

    pg_subpool_counts.carves++;
    if (++pg_subpool_counts.entries_live > pg_subpool_counts.entries_peak)
        pg_subpool_counts.entries_peak = pg_subpool_counts.entries_live;
    return i;
}

void *
pg_subpool_hand(unsigned int i)
{
    pg_subpool_counts.hands++;
    return sublet_take(&chunk_tab[i].slot);
}

void
pg_subpool_drop(unsigned int i)
{
    /* The alias the program holds dies here, and the slot holds the chunk
     * again, so the next hand-out of this entry needs no carve. That is why a
     * free does not release the entry: the entry is where the chunk lives
     * while it waits on a size-class list. */
    sublet_give(&chunk_tab[i].slot);
    pg_subpool_counts.drops++;
}

pg_chunk *
pg_subpool_entry(unsigned int i)
{
    return &chunk_tab[i];
}

pg_block *
pg_subpool_block_at(unsigned int i)
{
    return &block_tab[i];
}

unsigned int
pg_subpool_block_index(pg_block *b)
{
    return (unsigned int) (b - block_tab);
}

unsigned int
pg_subpool_first_chunk(pg_block *b)
{
    return b->chunk_head;
}

unsigned int
pg_subpool_next_chunk(unsigned int i)
{
    return chunk_tab[i].block_next;
}

void *
pg_subpool_header(unsigned long bytes)
{
    unsigned int i;

    if (bytes > PG_HEADER_BYTES)
        return NULL;
    if (header_free_n)
        i = header_free[--header_free_n];
    else if (header_next < PG_SUBPOOL_MAX)
        i = header_next++;
    else
        return NULL;
    for (unsigned long k = 0; k < PG_HEADER_BYTES; k++)
        header_tab[i][k] = 0;
    return &header_tab[i][0];
}

void
pg_subpool_header_free(void *p)
{
    unsigned char *q = (unsigned char *) p;
    unsigned int i;

    if (!q || header_free_n >= PG_SUBPOOL_MAX)
        return;
    /* Which slot, by subtracting one pointer from another inside the same
     * array. The difference is an integer and no capability is made from it,
     * which is what a cast to an integer and back would have done: the
     * compiler refuses that here, and it is right to. */
    i = (unsigned int) ((q - &header_tab[0][0]) / PG_HEADER_BYTES);
    if (i < PG_SUBPOOL_MAX)
        header_free[header_free_n++] = i;
}

const struct sublet_stats *
pg_subpool_primitives(void)
{
    return &sublet_stats;
}
