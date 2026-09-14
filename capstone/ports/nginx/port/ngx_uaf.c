/* Does a pointer into a destroyed pool still work? The one claim the whole port rests on, and the
 * one the 78-check driver cannot make, because a fault ends the domain and a domain that faulted
 * cannot report anything beside it. So it gets its own image, and the fault is the RESULT.
 *
 * THE TEST HAS TO DISCRIMINATE. A fault on its own proves nothing: a port can fault for any
 * number of reasons, and a test whose pass condition is "something went wrong" passes for the
 * wrong reasons too. So the same code runs over both levels below, and the two must disagree.
 *
 *   unprotected  pg_level0 hands out ordinary pointers. Destroying the pool returns the block to
 *                a free list and nothing else. The touch reads the byte that was written, which
 *                is the blindspot this paper is about, measured rather than asserted.
 *   under Sublet the block arrives linear, the object is carved from it, and destroy is one
 *                revocation. The touch must fault.
 *
 * And it has to reach the touch. NGX_UAF_STOP says how far to go, so a run that stops one step
 * short returns a mark and proves the setup is sound:
 *
 *   1  the pool is created, the object written and read back
 *   2  the pool is destroyed, nothing touched
 *   3  the object is touched after the destroy
 *   4  the same-sized pool is created again and lands on the same block
 *   5  the old pointer is read while a NEW object occupies its address
 *   6  the old pointer is offered to ngx_pfree of the pool that owns it now
 *
 * Stage 3 is the only one that may fault, and only in the protected arm. A fault at stage 1 or 2,
 * or in the unprotected arm at all, is a defect in the port and not a result.
 */
#include <stddef.h>
#include <stdint.h>

#include "ngx_shim.h"
#include "ngx_palloc.h"

#define NGX_DOM_MARK(n) do { *res = 0x4E000000u | ((unsigned) (n) & 0x00FFFFFFu); return; } while (0)

#ifndef NGX_UAF_STOP
#define NGX_UAF_STOP 3
#endif

#ifdef NGX_SUBLET
#include "ngx_subpool.h"
static sublet_cap arena_slot;
#else
void pg_level0_init(void *region, size_t bytes);
static unsigned char *arena_base;
static size_t arena_bytes;
#endif

void domain_main(unsigned *res, unsigned func);

/* Rung 0 returns &domain_main, masked to 32 bits, which is the convention
   capstone/tests/runtime-qemu/fault-locate.py needs to turn a fault pc into a symbol. Without it
   the load base has to be guessed, and a guessed base reads a fault into whatever function the
   arithmetic lands in. It read one into ngx_palloc_block+0x3fd68 here, an offset no function has,
   which is how the guess announced itself. */
static unsigned rungs;
#define NGX_ANCHOR_RUNG() do {                                                  \
    if (++rungs == 1) {                                                         \
        *res = (unsigned) (unsigned long) (void *) &domain_main;                \
        return;                                                                 \
    }                                                                           \
} while (0)

void domain_main(unsigned *res, unsigned func) {
    if (func == 1) {
#ifdef NGX_SUBLET
        sublet_store(&arena_slot, res);
#else
        unsigned long lo = (unsigned long) (void *) res;
        unsigned long hi = __builtin_capstone_cap_get_end((void *) res);
        arena_base = (unsigned char *) res;
        arena_bytes = (size_t) (hi - lo);
#endif
        return;
    }

    NGX_ANCHOR_RUNG();

#ifdef NGX_SUBLET
    if (sublet_type(&arena_slot) != 0) {
        NGX_DOM_MARK(0xFD0000u | (unsigned) sublet_type(&arena_slot));
    }
    ngx_subpool_init(&arena_slot);
#else
    if (arena_base == NULL) {
        NGX_DOM_MARK(0xFE0000u);
    }
    pg_level0_init(arena_base, arena_bytes);
#endif

    ngx_pool_t *pool = ngx_create_pool(1024, NULL);
    if (pool == NULL) {
        NGX_DOM_MARK(0xE10000u);
    }

    unsigned char *a = ngx_palloc(pool, 64);
    if (a == NULL) {
        NGX_DOM_MARK(0xE20000u);
    }

    for (int i = 0; i < 64; i++) {
        a[i] = (unsigned char) (0xA0 + i);
    }
    if (a[0] != 0xA0 || a[63] != (unsigned char) (0xA0 + 63)) {
        NGX_DOM_MARK(0xE30000u);      /* it never held what was written, so nothing below means anything */
    }

    if (NGX_UAF_STOP < 2) {
        NGX_DOM_MARK(0xC10000u);      /* alive and correct, before the destroy */
    }

    ngx_destroy_pool(pool);

    if (NGX_UAF_STOP < 3) {
        NGX_DOM_MARK(0xC20000u);      /* destroyed, nothing touched */
    }

    if (NGX_UAF_STOP == 3) {
        /* The touch. Under the discipline this is where the domain ends. Without it, the byte
           comes back and is reported, which is the whole point of the unprotected arm. */
        unsigned char v = a[0];
        NGX_DOM_MARK(0xC30000u | (unsigned) v);
    }

    /* ---- the same address again, which is a different question ------------
     *
     * Stage 3 asks what a stale pointer reads when nothing has taken its place. tab:safety asks
     * two more, and both need the address BACK in use:
     *
     *   4  same address, new object, old pointer. The level below keeps a free list per size, so
     *      a pool of the same size gets the same block. A stale read then returns the NEW
     *      object's bytes on the unprotected arm, which is the case a bounds check cannot see
     *      and a lifetime can.
     *   5  stale free after address reuse. The old pointer is offered to ngx_pfree of the pool
     *      that now owns that memory.
     *
     * The new pool writes a different pattern on purpose. A stale read that returned 0xA0 would
     * mean the memory was never reused and the scenario proved nothing, so the mark carries the
     * byte and the runner expects the NEW one.
     */
    ngx_pool_t *pool2 = ngx_create_pool(1024, NULL);
    if (pool2 == NULL) {
        NGX_DOM_MARK(0xE40000u);
    }

    unsigned char *b = ngx_palloc(pool2, 64);
    if (b == NULL) {
        NGX_DOM_MARK(0xE50000u);
    }

    for (int i = 0; i < 64; i++) {
        b[i] = (unsigned char) (0x5B + i);
    }

    /* Did the address actually come back? Read as an address only, never cast back. The answer
       is carried in the mark rather than assumed, because every stage after this one is empty
       if it is no. */
    unsigned same = ((unsigned long) (void *) a == (unsigned long) (void *) b) ? 1u : 0u;

    if (NGX_UAF_STOP < 5) {
        NGX_DOM_MARK(0xC40000u | same);
    }

    if (NGX_UAF_STOP == 5) {
        unsigned char v2 = a[0];      /* the old pointer, the new object */
        NGX_DOM_MARK(0xC50000u | (unsigned) v2);
    }

    /* A free through the stale pointer, offered to the pool that owns the memory now.
     *
     * THE POOL NEEDS A LARGE ALLOCATION FIRST, and the first version of this stage did not give
     * it one. ngx_pfree walks only the large list, comparing the pointer it was given against
     * each entry, so with an empty list the loop runs zero times and the stale pointer is never
     * touched at all. Both arms then answered NGX_DECLINED and the stage proved nothing. With an
     * entry there, the comparison actually happens, which is the question: whether offering a
     * revoked capability to an allocator that only COMPARES it is caught. */
    unsigned char *big = ngx_palloc(pool2, 8192);
    if (big == NULL) {
        NGX_DOM_MARK(0xE60000u);
    }
    big[0] = 0x7C;

    ngx_int_t rc = ngx_pfree(pool2, a);
    NGX_DOM_MARK(0xC60000u | ((unsigned) rc & 0xFFu));
}
