/* Replay what real nginx asked its pool for, in a domain.
 *
 * The driver in ngx_domain.c makes calls I chose. It proves the port answers its interface and it
 * cannot say what nginx asks for, so a cost measured against it would be a cost of my invention.
 * This one makes nginx's own calls, in nginx's own order, read from a trace the recorder in the
 * paper's experiments/a11/nginx took from a worker under wrk.
 *
 * WHAT IS REPLAYED AND WHAT IS NOT. Every call to the pool interface, with the sizes the program
 * asked for and in the order it asked. Not what the program did with the memory afterwards: the
 * trace records no write, and inventing one would put traffic in both arms that neither measured.
 * The cost here is the allocator's, which is the cost in question.
 *
 * THREE REGIONS, in the order the domain expects them:
 *   1  the trace, which the guest read from a file
 *   2  the arena the level below carves, LINEAR in the protected arm and plain in the other
 *   3  a scratch, holding the two identity tables and the result block
 *
 * THE RESULT DOES NOT FIT A RETURN VALUE. Two hundred thousand records cannot be reported in the
 * 24 bits a marker has, so the domain writes a block at the start of the scratch and the guest
 * reads it back. A marker still carries the reason when the replay refuses to start, because at
 * that point there is nothing in the scratch worth reading.
 */
#include <stddef.h>
#include <stdint.h>

#include "ngx_shim.h"
#include "ngx_palloc.h"
#include "ngxtrace.h"

#ifdef NGX_SUBLET
#include "ngx_subpool.h"
#else
void pg_level0_init(void *region, size_t bytes);
#endif

extern unsigned long ngx_level0_live;

#define NGX_DOM_MARK(n) do { *res = 0x4E000000u | ((unsigned) (n) & 0x00FFFFFFu); return; } while (0)

void domain_main(unsigned *res, unsigned func);

static unsigned rungs;
#define NGX_ANCHOR_RUNG() do {                                                  \
    if (++rungs == 1) {                                                         \
        *res = (unsigned) (unsigned long) (void *) &domain_main;                \
        return;                                                                 \
    }                                                                           \
} while (0)

/* ---- what the guest reads back, at the front of the scratch -------------- */
struct ngx_replay_out {
    uint64_t magic;          /* so the guest knows the domain got this far */
    uint64_t records;        /* in the file */
    uint64_t executed;
    uint64_t skipped;        /* a record naming a pool or an object this replay does not hold */
    uint64_t failures;       /* the allocator said no where the trace says it said yes */
    uint64_t ops[16];
    uint64_t pools_at_end;   /* still live when the trace ran out, and then destroyed */
    uint64_t objs_at_end;
    uint64_t level0_live;    /* what the level below still holds, which has to be zero */
    uint64_t carved, reused; /* the discipline's own counters, zero in the plain arm */
    uint64_t tables_full;    /* an identity table that ran out, which invalidates the run */
    /* What the discipline asked of the machine. The revocation node space is a fixed bump
       allocator of about a thousand with no reclamation, so these are not curiosities: they say
       whether a workload fits on the silicon at all, and SQLite's 183 carves did. */
    uint64_t n_split, n_mrev, n_delin, n_revoke, n_init;
};

/* ---- identities ----------------------------------------------------------
 * The recorder sees a whole process and assigns ids into the hundreds of thousands. This sees the
 * LIVE set, which the trace says is ten pools and six hundred objects at once, so the tables are
 * small and have to give entries back.
 *
 * Chained hashing over a free list, not open addressing. Deleting from an open-addressed table
 * needs either a tombstone, which fills a small table, or a backward shift, which has to fix up
 * every index that pointed at the moved entry. A chain is unlinked by walking it, and the chains
 * here are shorter than one entry on average.
 */
#define POOLBITS 8                        /* 256 buckets for a live set of ten */
#define OBJBITS  12                       /* 4096 for a live set of six hundred */
#define POOLMAX  512
#define OBJMAX   8192

struct poolent { uint32_t id; ngx_pool_t *p; int32_t hnext, objs, free; };
struct objent  { uint32_t id, pool; void *p; int32_t hnext, pnext, free; };

static struct poolent *pools;
static struct objent  *objs;
static int32_t *pbucket, *obucket;
static int32_t pfree_head, ofree_head;
static uint64_t tables_full;
static uint64_t pools_live, objs_live;

static uint32_t hash32(uint32_t x) { return (x * 2654435761u); }
#define PB(id) ((int32_t) (hash32(id) >> (32 - POOLBITS)))
#define OB(id) ((int32_t) (hash32(id) >> (32 - OBJBITS)))

static void tables_init(void) {
    for (int32_t i = 0; i < (1 << POOLBITS); i++) pbucket[i] = -1;
    for (int32_t i = 0; i < (1 << OBJBITS); i++) obucket[i] = -1;
    for (int32_t i = 0; i < POOLMAX; i++) { pools[i].id = 0; pools[i].free = i + 1; }
    pools[POOLMAX - 1].free = -1;
    for (int32_t i = 0; i < OBJMAX; i++) { objs[i].id = 0; objs[i].free = i + 1; }
    objs[OBJMAX - 1].free = -1;
    pfree_head = 0; ofree_head = 0;
}

static int32_t pool_find(uint32_t id) {
    for (int32_t i = pbucket[PB(id)]; i >= 0; i = pools[i].hnext) {
        if (pools[i].id == id) return i;
    }
    return -1;
}
static int32_t obj_find(uint32_t id) {
    for (int32_t i = obucket[OB(id)]; i >= 0; i = objs[i].hnext) {
        if (objs[i].id == id) return i;
    }
    return -1;
}

static int32_t pool_add(uint32_t id, ngx_pool_t *p) {
    if (pfree_head < 0) { tables_full++; return -1; }
    int32_t i = pfree_head; pfree_head = pools[i].free;
    pools[i].id = id; pools[i].p = p; pools[i].objs = -1;
    pools[i].hnext = pbucket[PB(id)]; pbucket[PB(id)] = i;
    pools_live++;
    return i;
}
static int32_t obj_add(uint32_t id, uint32_t pool, void *p) {
    if (ofree_head < 0) { tables_full++; return -1; }
    int32_t i = ofree_head; ofree_head = objs[i].free;
    objs[i].id = id; objs[i].pool = pool; objs[i].p = p;
    objs[i].hnext = obucket[OB(id)]; obucket[OB(id)] = i;
    int32_t pi = pool_find(pool);
    if (pi >= 0) { objs[i].pnext = pools[pi].objs; pools[pi].objs = i; }
    else { objs[i].pnext = -1; }
    objs_live++;
    return i;
}

static void obj_unhash(int32_t i) {
    int32_t *slot = &obucket[OB(objs[i].id)];
    while (*slot >= 0 && *slot != i) slot = &objs[*slot].hnext;
    if (*slot == i) *slot = objs[i].hnext;
    objs[i].id = 0;
    /* The capability goes too, and not for tidiness. A replay's own bookkeeping is the one thing
       in this image that holds capabilities to objects it no longer has any business with, and
       what a revocation tree counts is what is still reachable. Clearing the id alone would leave
       the port measured through a table the port does not have. */
    objs[i].p = NULL;
    objs[i].free = ofree_head; ofree_head = i;
    objs_live--;
}

/* One object, named by a PFREE. It has to leave its pool's list too, which is a walk of that
   pool's own objects and not of the table. */
static void obj_drop(int32_t i) {
    int32_t pi = pool_find(objs[i].pool);
    if (pi >= 0) {
        int32_t *slot = &pools[pi].objs;
        while (*slot >= 0 && *slot != i) slot = &objs[*slot].pnext;
        if (*slot == i) *slot = objs[i].pnext;
    }
    obj_unhash(i);
}

/* Every object of a pool, at a reset or a destroy. No record names them, because none died on its
   own: the pool took them all at once, which is the whole point of a pool. */
static void pool_drop_objs(int32_t pi) {
    int32_t i = pools[pi].objs;
    while (i >= 0) {
        int32_t next = objs[i].pnext;
        obj_unhash(i);
        i = next;
    }
    pools[pi].objs = -1;
}

static void pool_drop(int32_t pi) {
    pool_drop_objs(pi);
    int32_t *slot = &pbucket[PB(pools[pi].id)];
    while (*slot >= 0 && *slot != pi) slot = &pools[*slot].hnext;
    if (*slot == pi) *slot = pools[pi].hnext;
    pools[pi].id = 0;
    pools[pi].p = NULL;
    pools[pi].free = pfree_head; pfree_head = pi;
    pools_live--;
}

/* ---- the regions --------------------------------------------------------- */
static unsigned shares;
static unsigned char *trace_base; static size_t trace_bytes;
static unsigned char *scratch_base; static size_t scratch_bytes;
#ifdef NGX_SUBLET
static sublet_cap arena_slot;
#else
static unsigned char *arena_base; static size_t arena_bytes;
#endif

/* A region's end comes from its base by pointer arithmetic and never from a cast of the builtin's
   integer: a cast carries the address and no tag, and everything derived from it afterwards would
   be untagged. Two ports have paid for that lesson. */
static size_t region_bytes(unsigned *res) {
    unsigned long lo = (unsigned long) (void *) res;
    unsigned long hi = __builtin_capstone_cap_get_end((void *) res);
    return (size_t) (hi - lo);
}

/* ---- reading the file ----------------------------------------------------
 * Field by field out of the bytes, not by casting the region to a struct pointer. The recorder
 * writes packed little-endian on another machine and another compiler, and a struct laid over
 * those bytes would agree only by luck. recsize and the endian word are checked, which is what
 * they are in the file for.
 */
static uint32_t rd32(const unsigned char *p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}
static uint64_t rd64(const unsigned char *p) {
    return (uint64_t) rd32(p) | ((uint64_t) rd32(p + 4) << 32);
}

#define HEADSZ 80
#define RECSZ  40

void domain_main(unsigned *res, unsigned func) {
    if (func == 1) {
        switch (++shares) {
        case 1: trace_base = (unsigned char *) res; trace_bytes = region_bytes(res); break;
        case 2:
#ifdef NGX_SUBLET
            sublet_store(&arena_slot, res);
#else
            arena_base = (unsigned char *) res; arena_bytes = region_bytes(res);
#endif
            break;
        case 3: scratch_base = (unsigned char *) res; scratch_bytes = region_bytes(res); break;
        default: break;
        }
        return;
    }

    NGX_ANCHOR_RUNG();

    if (shares != 3) NGX_DOM_MARK(0xE00000u | shares);
    if (trace_bytes < HEADSZ) NGX_DOM_MARK(0xE10000u);

    /* The header, refused rather than trusted. */
    const unsigned char *h = trace_base;
    static const char magic[8] = { 'N','G','X','T','R','A','C','E' };
    for (int i = 0; i < 8; i++) if (h[i] != (unsigned char) magic[i]) NGX_DOM_MARK(0xE20000u | (unsigned) i);
    if (rd32(h + 8) != NGXT_VERSION) NGX_DOM_MARK(0xE30000u | rd32(h + 8));
    if (rd32(h + 12) != RECSZ)       NGX_DOM_MARK(0xE40000u | rd32(h + 12));
    if (rd64(h + 16) != NGXT_ENDIAN) NGX_DOM_MARK(0xE50000u);

    uint64_t nrec = (trace_bytes - HEADSZ) / RECSZ;

    /* The scratch: the result block first, then the two tables. */
    struct ngx_replay_out *out = (struct ngx_replay_out *) (void *) scratch_base;
    size_t off = (sizeof(struct ngx_replay_out) + 63u) & ~(size_t) 63u;
    size_t need = off
                + (size_t) POOLMAX * sizeof(struct poolent)
                + (size_t) OBJMAX * sizeof(struct objent)
                + (size_t) (1 << POOLBITS) * sizeof(int32_t)
                + (size_t) (1 << OBJBITS) * sizeof(int32_t);
    if (scratch_bytes < need) NGX_DOM_MARK(0xE60000u | (unsigned) (need >> 10));

    pools   = (struct poolent *) (void *) (scratch_base + off);            off += (size_t) POOLMAX * sizeof(struct poolent);
    objs    = (struct objent  *) (void *) (scratch_base + off);            off += (size_t) OBJMAX * sizeof(struct objent);
    pbucket = (int32_t *) (void *) (scratch_base + off);                   off += (size_t) (1 << POOLBITS) * sizeof(int32_t);
    obucket = (int32_t *) (void *) (scratch_base + off);

    for (size_t i = 0; i < sizeof *out; i++) ((unsigned char *) out)[i] = 0;
    tables_init();
    tables_full = pools_live = objs_live = 0;

#ifdef NGX_SUBLET
    if (sublet_type(&arena_slot) != 0) NGX_DOM_MARK(0xFD0000u | (unsigned) sublet_type(&arena_slot));
    ngx_subpool_init(&arena_slot);
#else
    pg_level0_init(arena_base, arena_bytes);
#endif

    uint64_t executed = 0, skipped = 0, failures = 0;
    uint64_t ops[16];
    for (int i = 0; i < 16; i++) ops[i] = 0;

    /* nrec is what the REGION holds, and the region is rounded up to a megabyte past the file.
       The guest zeroes the tail, and the format has no op 0, so the first zero record is the end
       of the file. Counting the padding as skipped records would have let a gate on
       executed plus skipped equalling the total pass while nine thousand records were nothing. */
    uint64_t seen = 0;
    for (uint64_t k = 0; k < nrec; k++) {
        const unsigned char *r = trace_base + HEADSZ + k * RECSZ;
        uint32_t op = rd32(r), poolid = rd32(r + 4), objid = rd32(r + 8), aux = rd32(r + 12);
        uint64_t s1 = rd64(r + 16);
        if (op == 0) break;
        seen++;
        if (op < 16) ops[op]++;

        int32_t pi = poolid ? pool_find(poolid) : -1;

        switch (op) {
        case NGXT_CREATE: {
            ngx_pool_t *p = ngx_create_pool((size_t) s1, NULL);
            if (p == NULL) { failures++; break; }
            if (pool_add(poolid, p) < 0) { ngx_destroy_pool(p); break; }
            executed++;
            break;
        }
        case NGXT_DESTROY:
            if (pi < 0) { skipped++; break; }
            ngx_destroy_pool(pools[pi].p);
            pool_drop(pi);
            executed++;
            break;
        case NGXT_RESET:
            if (pi < 0) { skipped++; break; }
            ngx_reset_pool(pools[pi].p);
            pool_drop_objs(pi);
            executed++;
            break;
        case NGXT_PALLOC:
        case NGXT_PNALLOC:
        case NGXT_PCALLOC:
        case NGXT_PMEMALIGN: {
            if (pi < 0) { skipped++; break; }
            void *p;
            if      (op == NGXT_PALLOC)  p = ngx_palloc(pools[pi].p, (size_t) s1);
            else if (op == NGXT_PNALLOC) p = ngx_pnalloc(pools[pi].p, (size_t) s1);
            else if (op == NGXT_PCALLOC) p = ngx_pcalloc(pools[pi].p, (size_t) s1);
            else                         p = ngx_pmemalign(pools[pi].p, (size_t) s1, (size_t) aux);
            /* The trace only holds calls that returned something, so a NULL here is the port
               refusing what nginx was given. */
            if (p == NULL) { failures++; break; }
            obj_add(objid, poolid, p);
            executed++;
            break;
        }
        case NGXT_PFREE: {
            if (pi < 0) { skipped++; break; }
            int32_t oi = obj_find(objid);
            if (oi < 0) { skipped++; break; }
            if (ngx_pfree(pools[pi].p, objs[oi].p) != NGX_OK) failures++;
            obj_drop(oi);
            executed++;
            break;
        }
        case NGXT_CLEANUP: {
            if (pi < 0) { skipped++; break; }
            ngx_pool_cleanup_t *c = ngx_pool_cleanup_add(pools[pi].p, (size_t) s1);
            if (c == NULL) { failures++; break; }
            obj_add(objid, poolid, c);
            executed++;
            break;
        }
        default:
            skipped++;
            break;
        }
    }

    out->records = seen;
    out->executed = executed;
    out->skipped = skipped;
    out->failures = failures;
    for (int i = 0; i < 16; i++) out->ops[i] = ops[i];
    out->pools_at_end = pools_live;
    out->objs_at_end = objs_live;

    /* A cut trace ends mid-request, so pools are still alive. They are destroyed here, because the
       level below has to come back to where it started and a leak has to be a leak of the port and
       not of the cut. */
    for (int32_t i = 0; i < POOLMAX; i++) {
        if (pools[i].id != 0) { ngx_destroy_pool(pools[i].p); pool_drop(i); }
    }

#ifdef NGX_SUBLET
    out->level0_live = ngx_subpool_live;
    out->carved = ngx_subpool_carved;
    out->reused = ngx_subpool_reused;
#else
    out->level0_live = ngx_level0_live;
#endif
    out->tables_full = tables_full;
#ifdef NGX_SUBLET
    out->n_split = sublet_stats.split;
    out->n_mrev = sublet_stats.mrev;
    out->n_delin = sublet_stats.delin;
    out->n_revoke = sublet_stats.revoke;
    out->n_init = sublet_stats.init;
#endif
    out->magic = 0x4E47585245504CAull;   /* written LAST: a partial block cannot read as a result */

    NGX_DOM_MARK(0xC00000u);
}
