/* Pool observations shared by native recording and the source-level replay.
 * Uses integer addresses only for local identity lookup. Addresses never go
 * on the wire, and the observer never supplies capabilities to the allocator.
 */
#include <string.h>
#include "trace.h"
struct observed_pool { uintptr_t address; uint64_t size, flags, kind; unsigned closed; };
struct observed_block {
    uintptr_t address;
    uint64_t pool, size, object, freed_at;
    unsigned alive, live, used;
};
static struct observed_pool pools[FF2_POOLS];
static struct observed_block blocks[FF2_BLOCKS];
static struct ff2_event stack[128];
static uint64_t npools, nblocks, nobjects, ncalls, depth;
static uint64_t allocations[3], live[3], retained[3];

void ff2_reset(void)
{
    memset(pools, 0, sizeof pools); memset(blocks, 0, sizeof blocks);
    memset(stack, 0, sizeof stack);
    memset(allocations, 0, sizeof allocations);
    memset(live, 0, sizeof live); memset(retained, 0, sizeof retained);
    npools = nblocks = nobjects = ncalls = depth = 0;
}
static uint64_t pool_id(unsigned kind, void *p)
{
    for (uint64_t i = npools; i; i--)
        if (pools[i].kind == kind && pools[i].address == (uintptr_t)p) return i;
    ff2_fail(101);
}
static uint64_t block_id(uint64_t pool, void *p)
{
    for (uint64_t i = nblocks; i; i--)
        if (blocks[i].alive && blocks[i].pool == pool && blocks[i].address == (uintptr_t)p)
            return i;
    ff2_fail(102);
}
static void emit(struct ff2_event e)
{
    e.allocations = allocations[e.kind];
    e.live = live[e.kind]; e.retained = retained[e.kind];
    ff2_sink(&e);
}
uint64_t ff2_begin(unsigned op, unsigned kind, void *p, size_t size,
                   uint64_t flags, void *object)
{
    if (kind < 1 || kind > 2 || depth == 128) ff2_fail(103);
    struct ff2_event e = {0};
    e.op = op; e.kind = kind; e.call = ++ncalls;
    e.parent = depth ? stack[depth - 1].call : 0;
    if (op == FF2_CREATE) {
        if (++npools >= FF2_POOLS) ff2_fail(104);
        e.pool = npools; e.size = size; e.flags = flags;
        pools[npools].kind = kind; pools[npools].size = size; pools[npools].flags = flags;
    } else {
        e.pool = pool_id(kind, p);
        e.size = pools[e.pool].size;
        if (op == FF2_GET) {
            if (pools[e.pool].closed) ff2_fail(105);
            e.object = ++nobjects;
        } else if (op == FF2_RETURN) {
            e.backing = block_id(e.pool, object);
            struct observed_block *b = &blocks[e.backing];
            if (!b->live) ff2_fail(106);
            e.object = b->object;
            b->live = 0; b->freed_at = allocations[kind];
            live[kind] -= b->size;
        } else if (op == FF2_CLOSE) {
            if (pools[e.pool].closed) ff2_fail(107);
            pools[e.pool].closed = 1;
        } else if (op == FF2_CALLBACK) {
            e.flags = flags;
            if (object) {
                e.backing = block_id(e.pool, object);
                e.object = blocks[e.backing].object;
            }
        } else ff2_fail(108);
    }
    stack[depth++] = e;
    emit(e);
    return e.call;
}
void ff2_end(uint64_t call, void *result)
{
    if (!depth || stack[depth - 1].call != call) ff2_fail(109);
    struct ff2_event e = stack[--depth];
    if (e.op == FF2_CREATE) {
        if (!result) ff2_fail(110);
        pools[e.pool].address = (uintptr_t)result;
    } else if (e.op == FF2_GET) {
        if (!result) ff2_fail(111);
        e.backing = block_id(e.pool, result);
        struct observed_block *b = &blocks[e.backing];
        if (b->live) ff2_fail(112);
        allocations[e.kind]++;
        e.gap = b->used ? allocations[e.kind] - b->freed_at : 0;
        b->used = b->live = 1; b->object = e.object;
        live[e.kind] += b->size;
    }
    e.op |= FF2_FINISH;
    emit(e);
}
void ff2_new(unsigned kind, void *p, void *obj)
{
    if (!depth || ++nblocks >= FF2_BLOCKS) ff2_fail(113);
    uint64_t pi = pool_id(kind, p);
    struct observed_block *b = &blocks[nblocks];
    b->address = (uintptr_t)obj; b->pool = pi; b->size = pools[pi].size;
    b->alive = 1; b->object = stack[depth - 1].object;
    retained[kind] += b->size;
    struct ff2_event e = {0};
    e.op = FF2_NEW; e.kind = kind; e.pool = pi; e.parent = stack[depth - 1].call;
    e.object = b->object; e.size = b->size; e.backing = nblocks;
    emit(e);
}
void ff2_drop(unsigned kind, void *p, void *obj)
{
    uint64_t pi = pool_id(kind, p), bi = block_id(pi, obj);
    struct observed_block *b = &blocks[bi];
    if (b->live) ff2_fail(114);
    retained[kind] -= b->size; b->alive = 0;
    struct ff2_event e = {0};
    e.op = FF2_DROP; e.kind = kind; e.pool = pi; e.parent = depth ? stack[depth - 1].call : 0;
    e.object = b->object; e.size = b->size; e.backing = bi;
    emit(e);
}
uint64_t ff2_callback(unsigned kind, void *p, void *obj, unsigned type)
{ return ff2_begin(FF2_CALLBACK, kind, p, 0, type, obj); }
void ff2_finish(void)
{
    if (depth) ff2_fail(115);
    for (uint64_t i = 1; i <= npools; i++) if (!pools[i].closed) ff2_fail(116);
    if (live[1] || live[2] || retained[1] || retained[2]) ff2_fail(117);
    emit((struct ff2_event){.op = FF2_DONE});
}
