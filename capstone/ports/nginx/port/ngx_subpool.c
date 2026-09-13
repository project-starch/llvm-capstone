/* The level below nginx's pool, under Sublet: blocks that arrive LINEAR, so the pool above can
 * carve objects out of them and one revocation can take a whole pool's objects with it.
 *
 * WHY A SECOND LEVEL 0 AND NOT pg_level0.c. That one is a real allocator over a region and it is
 * what the unprotected arm uses, deliberately, so that a difference between the arms is never the
 * level below being different in kind. What it cannot do is hand out a LINEAR block: it returns
 * ordinary pointers into one arena, and csmrev refuses anything that is not CAP_TYPE_LIN, so
 * nothing carved from such a block could ever be revoked. This file is the same allocator with
 * that one difference.
 *
 * IT NEVER MERGES, and it does not need to. nginx asks for blocks of a handful of distinct sizes,
 * the pool size it was created with, and gives each one back whole at ngx_destroy_pool. So a
 * freed block goes on a free list for its own size and is handed out again as it is. That is the
 * property the MicroPython collector did not have and could not be given: there, free space came
 * back one object at a time and had to be joined, and joining needs a handle senior to exactly two
 * neighbours, which a front-carving allocator never has.
 *
 * The arena arrives once, as one linear region, and is carved forward. What is carved is never
 * returned to it -- only to the free list -- so the arena's own cursor only moves one way.
 */
#include <stddef.h>
#include <stdint.h>

#include "sublet.h"

#ifndef NGX_SUBPOOL_SIZES
#define NGX_SUBPOOL_SIZES 8       /* distinct block sizes; nginx uses one or two */
#endif
#ifndef NGX_SUBPOOL_FREE_PER_SIZE
#define NGX_SUBPOOL_FREE_PER_SIZE 64
#endif

/* One entry per free block: the region itself, linear, ready to be handed out again. */
struct free_list {
    size_t bytes;                                   /* what this list's blocks measure */
    unsigned count;
    sublet_cap slot[NGX_SUBPOOL_FREE_PER_SIZE];
};

static sublet_cap arena;                            /* what is left of the region, linear */
size_t ngx_subpool_arena_left;                      /* and how much of it, counted rather than asked */
static struct free_list lists[NGX_SUBPOOL_SIZES];
static unsigned n_lists;

/* Counters, so the test and the measurement can both ask what happened rather than infer it. */
unsigned long ngx_subpool_carved;                   /* blocks cut from the arena */
unsigned long ngx_subpool_reused;                   /* blocks handed out from a free list */
unsigned long ngx_subpool_returned;                 /* blocks given back */
unsigned long ngx_subpool_live;                     /* handed out and not yet back */

void ngx_subpool_init(sublet_cap *region) {
    sublet_move(region, &arena);
    ngx_subpool_arena_left = (size_t) (sublet_end(&arena) - sublet_base(&arena));
    n_lists = 0;
    ngx_subpool_carved = ngx_subpool_reused = ngx_subpool_returned = ngx_subpool_live = 0;
    for (unsigned i = 0; i < NGX_SUBPOOL_SIZES; i++) {
        lists[i].bytes = 0;
        lists[i].count = 0;
    }
}

static struct free_list *list_for(size_t bytes, int create) {
    for (unsigned i = 0; i < n_lists; i++) {
        if (lists[i].bytes == bytes) {
            return &lists[i];
        }
    }
    if (!create || n_lists == NGX_SUBPOOL_SIZES) {
        return NULL;
    }
    lists[n_lists].bytes = bytes;
    lists[n_lists].count = 0;
    return &lists[n_lists++];
}

/* A block, linear, with its handle. The caller keeps both: the region to carve from and the
   handle that will revoke everything carved out of it. */
int ngx_subpool_block(size_t bytes, sublet_cap *region, sublet_cap *handle) {
    /* A REGION HAS TO BE A WHOLE NUMBER OF CAPABILITIES LONG. sublet.h says so where it explains
       the give: a region that is not cannot be written through, because the fill loop stops at
       the last whole capability and the init behind it traps. memsys5 never trips over this,
       because it hands out powers of two of a 64 byte atom. nginx does: six of its blocks over
       two hundred thousand calls are 4280 bytes, and one such carve leaves the arena's base
       8 aligned, so EVERY later block is misaligned and the first capability stored in one is an
       unaligned access. The synthetic driver never asked for an odd size. */
    bytes = (bytes + 15) & ~(size_t) 15;

    struct free_list *fl = list_for(bytes, 0);
    if (fl != NULL && fl->count > 0) {
        sublet_move(&fl->slot[--fl->count], region);
        ++ngx_subpool_reused;
    } else {
        /* Counted, not asked. A carve that reaches exactly to the end of a region moves the whole
           of it out and leaves the slot EMPTY, and asking an empty slot for its base is an
           untagged operand, cause 24. So the arena that is exactly spent would have faulted on
           the request after the one that spent it, which is the opposite of refusing. A count can
           be read when a slot cannot. */
        if (ngx_subpool_arena_left < bytes) {
            return 0;                               /* the arena is spent */
        }
        sublet_carve(&arena, sublet_base(&arena) + bytes, region);
        ngx_subpool_arena_left -= bytes;
        ++ngx_subpool_carved;
    }
    /* Senior to everything the level above will carve out of this block, taken BEFORE the first
       carve, because that is the only moment at which such a handle can be had. */
    sublet_handle(region, handle);
    ++ngx_subpool_live;
    return 1;
}

/* One revocation, and every object the level above carved out of this block dies with it. The
   block itself comes back whole and goes on its size's free list.

   The size is NOT a parameter. A block that has come back is whole again by definition, so it
   measures itself, and the caller above cannot file a block on the wrong list by passing a size
   that no longer matches. nginx's pool would otherwise have had to carry the block size along its
   whole chain to get it back here. */
void ngx_subpool_release(sublet_cap *region, sublet_cap *handle) {
    sublet_give_to(handle, region);
    ++ngx_subpool_returned;
    --ngx_subpool_live;
    size_t bytes = (size_t) (sublet_end(region) - sublet_base(region));
    struct free_list *fl = list_for(bytes, 1);
    if (fl == NULL || fl->count == NGX_SUBPOOL_FREE_PER_SIZE) {
        sublet_clear(region);                       /* nowhere to keep it: let it go */
        return;
    }
    sublet_move(region, &fl->slot[fl->count++]);
}
