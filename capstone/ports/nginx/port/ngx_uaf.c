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
 *   7  an ancestor revokes while a nested handle is alive, and the nested handle is asked what
 *      it is afterwards. No unprotected arm: a level below that hands out ordinary pointers has
 *      no nested authority to end
 *   8  the same block is then re-issued, to a new handle and a new object, so that the stale
 *      handle stage 9 offers back is a handle to memory somebody else owns
 *   9  that stale handle is offered back, which is the operation a double free is made of
 *
 * Stages 3, 5 and 9 are the ones that may fault, and only in the protected arm. A fault at stage
 * 1 or 2, or in the unprotected arm at all, is a defect in the port and not a result.
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
    if (NGX_UAF_STOP < 7) {
        NGX_DOM_MARK(0xC60000u | ((unsigned) rc & 0xFFu));
    }

#ifdef NGX_SUBLET
    /* ---- does an ancestor's revoke end a nested authority ------------------
     *
     * tab:safety's last hierarchy row. Scenario 7 of the level below's own test already shows
     * that the ancestor takes back the TERRITORY a nested handle governed, and stage 3 above
     * shows that an object carved under that handle is dead. What neither shows is what becomes
     * of the nested handle ITSELF.
     *
     * This stage has no unprotected arm and cannot have one: a level below that hands out
     * ordinary pointers has no nested authority to end. It is a question about the mechanism
     * rather than a difference between two arms, and the mark carries the answer.
     */
    sublet_cap nblk, nout, nin, nobj;
    if (!ngx_subpool_block(2048, &nblk, &nout)) {
        NGX_DOM_MARK(0xE70000u);
    }
    sublet_handle(&nblk, &nin);                  /* the nested authority */
    sublet_carve(&nblk, sublet_base(&nblk) + 64, &nobj);
    unsigned char *nd = (unsigned char *) sublet_take(&nobj);
    if (nd == NULL) {
        NGX_DOM_MARK(0xE80000u);
    }
    nd[0] = 0x9E;

    unsigned before = (unsigned) sublet_type(&nin);   /* REV while it is alive */

    sublet_give_to(&nout, &nblk);                     /* the ancestor revokes */

    /* What the nested handle is now. Reading its slot is itself the question: if the revoke
       invalidated it, this is where the domain ends, and that is an answer. If it comes back
       with a type, the mark says which. */
    unsigned after = (unsigned) sublet_type(&nin);

    /* The control, without which `after` is a number and not a result: what a slot that is KNOWN
       to hold nothing reports. If the two agree, the ancestor's revoke left the nested handle
       holding no capability, which is the row's answer. Valid types are 0 to 5. */
    sublet_cap empty;
    sublet_clear(&empty);
    unsigned none = (unsigned) sublet_type(&empty);

    if (NGX_UAF_STOP < 8) {
        NGX_DOM_MARK(0xC70000u | ((before & 0xFu) << 8) | ((after & 0xFu) << 4) | (none & 0xFu));
    }

    /* ---- a retained release handle, offered back ---------------------------
     *
     * The hierarchy test in the paper asks what an allocator that KEPT a release handle across
     * an ancestor's revoke can still do with it. Under this discipline it cannot keep one at
     * all, and that is worth measuring rather than arguing: a revocation capability is linear,
     * so it lives in exactly one slot and a copy would null the original. Stage 7 shows the
     * revoke empties that one slot. This stage asks what using it afterwards does.
     *
     * AND IT ONLY HAS TEETH IF THE SPACE HAS AN OWNER AGAIN. Offering back a handle to memory
     * nobody wants proves nothing either way, so the block is re-issued first: a new handle, a
     * new object, a byte written and read back. Stage 8 reports that the re-issue worked and
     * that the stale handle is still empty. Stage 9 then makes the offer. Were it to revoke,
     * it would take the new owner's object with it.
     */
    sublet_cap sblk, sout, sstale, sobj;
    if (!ngx_subpool_block(2048, &sblk, &sout)) {
        NGX_DOM_MARK(0xE90000u);
    }
    sublet_handle(&sblk, &sstale);
    sublet_carve(&sblk, sublet_base(&sblk) + 64, &sobj);
    unsigned char *sd = (unsigned char *) sublet_take(&sobj);
    if (sd == NULL) {
        NGX_DOM_MARK(0xEA0000u);
    }
    sd[0] = 0x8A;

    sublet_give_to(&sout, &sblk);                /* the ancestor revokes, and sstale is emptied */

    /* the space gets a new owner */
    sublet_cap sin2, sobj2;
    sublet_handle(&sblk, &sin2);
    sublet_carve(&sblk, sublet_base(&sblk) + 64, &sobj2);
    unsigned char *sd2 = (unsigned char *) sublet_take(&sobj2);
    if (sd2 == NULL) {
        NGX_DOM_MARK(0xEB0000u);
    }
    sd2[0] = 0xD1;

    unsigned stale = (unsigned) sublet_type(&sstale);
    unsigned live = (sd2[0] == 0xD1) ? 1u : 0u;

    if (NGX_UAF_STOP < 9) {
        NGX_DOM_MARK(0xC80000u | ((stale & 0xFu) << 4) | live);
    }

    /* The offer. If the domain ends here, the stale handle could not be replayed. If it returns,
       the mark says what landed in the destination slot and whether the new owner's byte is
       still there, because a revoke that succeeded would have taken it. */
    sublet_cap sdst;
    sublet_clear(&sdst);
    sublet_give_to(&sstale, &sdst);

    NGX_DOM_MARK(0xC90000u | (((unsigned) sublet_type(&sdst) & 0xFu) << 4)
                 | ((sd2[0] == 0xD1) ? 1u : 0u));
#else
    /* No nested authority exists on this arm, and the mark says so rather than leaving a gap.
       Stages 7 to 9 are all about authority the unprotected level below never had. */
    NGX_DOM_MARK(0xC70000u + (((unsigned) NGX_UAF_STOP - 7u) << 16) + 0xFF00u);
#endif
}
