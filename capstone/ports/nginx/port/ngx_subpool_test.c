/* Does the level below keep its promise? A domain that asks the hardware, before nginx is patched.
 *
 * The claim this port makes is that a pool's whole contents die in one revocation.
 * port/ngx_subpool.c is the piece that has to make that true, and it is the piece to test first,
 * because a fault inside a patched ngx_palloc.c would be much harder to read back to its cause.
 * The PostgreSQL port tests its own level below the same way and for the same reason.
 *
 * Thirteen scenarios, each a claim the design makes, numbered in the order they run:
 *
 *   1  a block comes out linear, objects carve from it, and what is written through an object
 *      reads back
 *   2  release is one revocation, and the block comes back WHOLE: the cursor is at the block's
 *      own start again, so the objects carved above are gone from it and not merely forgotten.
 *      Counters cannot tell those two apart, which is why the bases are compared
 *   3  the next request of that size REUSES the block rather than carving a new one, which is the
 *      property that lets an arena serve a thousand pools
 *   4  a different size carves a new block instead of reusing the wrong one
 *   5  a hundred cycles leave nothing outstanding
 *   6  an arena that is spent refuses rather than returning something that looks like a block
 *   7  a SECOND handle, taken on what is left of a block after something was carved off its
 *      front, revokes only that remainder, and the first handle still takes the whole block back
 *      afterwards. nginx's ngx_reset_pool needs exactly this and cannot be written without it:
 *      the pool header sits at the front of its own block and the caller keeps pointing at it
 *      across a reset, so a reset that revoked the whole block would invalidate the pointer its
 *      caller is about to use again
 *   8  a block of a size that is not a whole number of capabilities is rounded up rather than
 *      carved as asked. A carve of 4280 bytes leaves the arena 8 aligned, and from then on every
 *      block is misaligned and the first capability stored in one faults. Real nginx asks for
 *      4280; no driver written here ever did
 *   9  two thousand objects taken with at most twenty alive: a block's revocation takes back the
 *      nodes of the objects carved out of it, or the run does not reach the end
 *  10  six hundred objects alive AT ONCE out of ten blocks, which is the shape real nginx has.
 *      Nine and ten together say whether the revocation node ceiling is on what is alive or on
 *      what has ever been taken, and a replay of real traffic hit that ceiling with neither
 *      question answered
 *  11  three thousand cycles of a block taken with an outer handle, a header carved off its
 *      front and an inner handle taken behind it, which is exactly what the port does per pool.
 *      nginx creates fourteen thousand pools where the pool driver's balance scenario creates a
 *      thousand and passes, so handle churn was the third candidate for the ceiling a replay of
 *      real traffic hit. It is not the ceiling either: this passes
 *  12  two children of one block, each a region with a handle of its own, and one of them
 *      revokes its own authority. The other child's objects still read and still write, and the
 *      revoked child's region comes back to ITS bounds and not to the block's, which is what
 *      makes a revoke downward rather than sideways
 *  13  an arena spent to the LAST BYTE still refuses the next request. Not the same claim as 6:
 *      a carve that reaches exactly to the end of a region moves the whole of it out and leaves
 *      the slot empty, so the request after the one that spent the arena would have asked an
 *      empty slot for its base, which is a fault and not a refusal
 *
 * A phase counter rides in bits 16..23 of the result. It is not decoration: a build that silently
 * kept an older image, or a run that read an older log, reported a plausible count and no failures
 * three times in a row here. A phase that does not reach its last value says which of the two.
 *
 * NOT tested here: that an alias carved from a released block faults on the next touch. It does,
 * and it is the point, but a fault ends the domain and a domain that faulted cannot report the
 * five results beside it. That question wants its own image.
 */
#include <stddef.h>
#include <stdint.h>

#include "sublet.h"

#define NGX_DOM_MARK(n) do { *res = 0x4E000000u | ((unsigned) (n) & 0x00FFFFFFu); return; } while (0)

void ngx_subpool_init(sublet_cap *region);
int ngx_subpool_block(size_t bytes, sublet_cap *region, sublet_cap *handle);
void ngx_subpool_release(sublet_cap *region, sublet_cap *handle);
extern unsigned long ngx_subpool_carved, ngx_subpool_reused, ngx_subpool_returned,
                     ngx_subpool_live;
extern size_t ngx_subpool_arena_left;

static unsigned ran, failures, phase, failphase;
/* The first phase that failed, because a count says how many and not which, and the
   scenarios here exist to tell two explanations apart. */
#define CHECK(cond) do { ++ran; if (!(cond)) { ++failures; if (!failphase) failphase = phase; } } while (0)

/* Carve one object out of a block and write through it, the way the pool above will. */
static int carve_write_read(sublet_cap *region, size_t bytes, unsigned char seed) {
    sublet_cap slot;
    unsigned long base = sublet_base(region);
    if (sublet_end(region) - base < bytes) {
        return 0;
    }
    sublet_carve(region, base + bytes, &slot);
    unsigned char *p = (unsigned char *) sublet_take(&slot);
    if (p == NULL) {
        return 0;
    }
    for (size_t i = 0; i < bytes; i++) {
        p[i] = (unsigned char) (seed + i);
    }
    for (size_t i = 0; i < bytes; i++) {
        if (p[i] != (unsigned char) (seed + i)) {
            return 0;
        }
    }
    return 1;
}

static sublet_cap arena_slot;

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
        /* The arena, and it must arrive LINEAR or nothing below can ever be revoked. The type is
           checked rather than assumed: a REV_SHARED share arrives NONLIN, csmrev refuses it, and
           the failure would otherwise show up three calls later as something else. */
        sublet_store(&arena_slot, res);
        return;
    }

    NGX_ANCHOR_RUNG();

    if (sublet_type(&arena_slot) != 0) {         /* 0 = CAP_TYPE_LIN */
        NGX_DOM_MARK(0xFD0000u | (unsigned) sublet_type(&arena_slot));
    }

    ngx_subpool_init(&arena_slot);
    ran = failures = phase = failphase = 0;

    sublet_cap r1, h1, r2, h2;

    phase = 1;
    CHECK(ngx_subpool_block(1024, &r1, &h1) == 1);
    unsigned long b0 = sublet_base(&r1), e0 = sublet_end(&r1);
    CHECK(e0 - b0 == 1024);
    CHECK(carve_write_read(&r1, 64, 0x11));
    CHECK(carve_write_read(&r1, 64, 0x22));
    CHECK(sublet_base(&r1) == b0 + 128);        /* the two objects really left the block */
    CHECK(ngx_subpool_carved == 1 && ngx_subpool_live == 1);

    phase = 2;
    ngx_subpool_release(&r1, &h1);
    CHECK(ngx_subpool_returned == 1 && ngx_subpool_live == 0);

    /* The same size again: reused, not carved. And WHOLE, which is the claim the port rests on:
       the cursor is back at the block's own start, so the two objects carved above are gone from
       it, not merely forgotten. Counters could not tell these apart. */
    phase = 3;
    CHECK(ngx_subpool_block(1024, &r1, &h1) == 1);
    CHECK(sublet_base(&r1) == b0 && sublet_end(&r1) == e0);
    CHECK(ngx_subpool_reused == 1 && ngx_subpool_carved == 1);
    CHECK(carve_write_read(&r1, 64, 0x33));

    /* A different size: carved, not the wrong block reused. */
    phase = 4;
    CHECK(ngx_subpool_block(4096, &r2, &h2) == 1);
    CHECK(ngx_subpool_carved == 2);
    CHECK(carve_write_read(&r2, 128, 0x44));

    ngx_subpool_release(&r2, &h2);
    ngx_subpool_release(&r1, &h1);
    CHECK(ngx_subpool_live == 0);

    /* A hundred cycles, nothing outstanding at the end and no new carves after the first of each
       size, which is what reuse means. */
    phase = 5;
    unsigned long carved_before = ngx_subpool_carved;
    for (int i = 0; i < 100; i++) {
        if (!ngx_subpool_block(1024, &r1, &h1)) {
            ++failures;
            break;
        }
        if (!carve_write_read(&r1, 32, (unsigned char) i)) {
            ++failures;
        }
        ngx_subpool_release(&r1, &h1);
    }
    ++ran;
    if (ngx_subpool_live != 0 || ngx_subpool_carved != carved_before) {
        ++failures;
    }

    /* A block larger than anything left: refused, and nothing outstanding because of it. */
    phase = 6;
    sublet_cap r3, h3;
    CHECK(ngx_subpool_block((size_t) 1 << 30, &r3, &h3) == 0);
    CHECK(ngx_subpool_live == 0);

    /* Two handles on one block, the second junior to the first. Asked of the hardware here rather
       than assumed in the port, because the port's reset is built on the answer. */
    phase = 7;
    sublet_cap blk, outer, inner, hdrslot;
    CHECK(ngx_subpool_block(2048, &blk, &outer) == 1);
    unsigned long bb = sublet_base(&blk), be = sublet_end(&blk);

    sublet_carve(&blk, bb + 64, &hdrslot);          /* the pool header, off the front */
    unsigned char *hdr = (unsigned char *) sublet_take(&hdrslot);
    CHECK(hdr != NULL);
    hdr[0] = 0x5A;

    sublet_handle(&blk, &inner);                    /* senior to what follows, junior to outer */
    CHECK(carve_write_read(&blk, 128, 0x66));
    CHECK(sublet_base(&blk) == bb + 64 + 128);

    sublet_give_to(&inner, &blk);                   /* a reset */
    CHECK(sublet_base(&blk) == bb + 64 && sublet_end(&blk) == be);
    CHECK(hdr[0] == 0x5A);                          /* the header outlived the reset */

    sublet_give_to(&outer, &blk);                   /* a destroy */
    CHECK(sublet_base(&blk) == bb && sublet_end(&blk) == be);

    /* Taken back by hand rather than through release, which has already had its give_to done, so
       this block is dropped and the live counter is expected to still show it. */
    sublet_clear(&blk);
    CHECK(ngx_subpool_live == 1);

    /* Spend the arena to the last byte, then ask again. Before the count replaced the query this
       faulted with cause 24 instead of refusing, and scenario 6 could not see it: 6 asks for more
       than is left, which the size test catches, and this asks for exactly what is left. */
    /* An odd size, before the arena is spent, because what it breaks is everything AFTER it. */
    phase = 8;
    sublet_cap ro, rh2;
    unsigned long before_left = ngx_subpool_arena_left;
    CHECK(ngx_subpool_block(4280, &ro, &rh2) == 1);
    CHECK((sublet_end(&ro) - sublet_base(&ro)) == 4288);       /* rounded, not as asked */
    CHECK((ngx_subpool_arena_left % 16) == 0);
    CHECK(before_left - ngx_subpool_arena_left == 4288);
    ngx_subpool_release(&ro, &rh2);
    /* and the next block still starts where a capability may be stored */
    sublet_cap rn, hn;
    CHECK(ngx_subpool_block(64, &rn, &hn) == 1);
    CHECK((sublet_base(&rn) % 16) == 0);
    CHECK(carve_write_read(&rn, 64, 0xBB));
    ngx_subpool_release(&rn, &hn);

    /* Does a block's revocation take back the nodes of the objects carved out of it? Two
       thousand takes with at most twenty alive. The revocation node pool is a fixed bump
       allocator of about a thousand, so if a take's handle were not reclaimed when its block goes
       back, this would not reach the end. */
    phase = 9;
    {
        int ok = 1;
        for (int cyc = 0; cyc < 100 && ok; cyc++) {
            sublet_cap rc, hc;
            /* Marked on the spot rather than counted, because a scenario meant to tell two
               explanations apart has to say WHERE it stopped. Payload: 0x09 then the cycle, then
               1 if the block was refused and 2 if an object was. */
            if (!ngx_subpool_block(2048, &rc, &hc)) NGX_DOM_MARK(0x090000u | ((unsigned) cyc << 4) | 1u);
            for (int i = 0; i < 20; i++) {
                if (!carve_write_read(&rc, 64, (unsigned char) (cyc + i))) {
                    NGX_DOM_MARK(0x090000u | ((unsigned) cyc << 4) | 2u);
                }
            }
            ngx_subpool_release(&rc, &hc);
        }
        CHECK(ok);
        CHECK(ngx_subpool_live == 1);   /* 7 dropped one block by hand and says so */
    }

    /* And the other half of the question: six hundred objects alive AT ONCE out of ten blocks,
       which is the shape a trace of real nginx has. If the ceiling is on what is alive rather than
       on what has ever been taken, this is where it shows and the scenario above does not. */
    phase = 10;
    {
        sublet_cap blk[10], hnd[10];
        int nb = 0, ok = 1, taken = 0;
        for (; nb < 10; nb++) {
            if (!ngx_subpool_block(4096, &blk[nb], &hnd[nb])) { ok = 0; break; }
        }
        for (int i = 0; i < nb && ok; i++) {
            for (int j = 0; j < 60; j++) {
                if (!carve_write_read(&blk[i], 64, (unsigned char) j)) { ok = 0; break; }
                taken++;
            }
        }
        CHECK(ok);
        CHECK(taken == nb * 60);
        for (int i = 0; i < nb; i++) ngx_subpool_release(&blk[i], &hnd[i]);
        CHECK(ngx_subpool_live == 1);
    }

    /* Neither of those reached a ceiling, so the remaining difference between them and a replay
       of real nginx is the number of HANDLES. The port takes two per pool, one senior to the
       block and one behind the header, and nginx creates fourteen thousand pools where the
       driver's own balance scenario creates one thousand and passes. Three thousand cycles of
       exactly that shape, and if the ceiling is handle churn this is where it shows. */
    phase = 11;
    {
        for (int cyc = 0; cyc < 3000; cyc++) {
            sublet_cap rb, ho, hi;
            if (!ngx_subpool_block(1024, &rb, &ho)) NGX_DOM_MARK(0x0B0000u | ((unsigned) cyc << 4) | 1u);
            sublet_carve(&rb, sublet_base(&rb) + 128, &hi);   /* a header, as the port carves one */
            if (sublet_take(&hi) == NULL)          NGX_DOM_MARK(0x0B0000u | ((unsigned) cyc << 4) | 2u);
            sublet_handle(&rb, &hi);                          /* the inner handle, behind it */
            if (!carve_write_read(&rb, 64, (unsigned char) cyc))
                                                   NGX_DOM_MARK(0x0B0000u | ((unsigned) cyc << 4) | 3u);
            ngx_subpool_release(&rb, &ho);                    /* the outer takes all of it back */
        }
        ++ran;
        if (ngx_subpool_live != 1) { ++failures; if (!failphase) failphase = phase; }
    }

    /* Two children of one block, and one of them is uncooperative.
     *
     * tab:safety calls this "sibling survives uncooperative child". A child here is a region of
     * the block with a handle of its own, which is what a sub-pool is. The uncooperative act is
     * revoking that handle without telling anyone, and what has to hold is that the OTHER child's
     * objects are untouched by it. A discipline that gave the two children overlapping authority,
     * or that let a revoke walk sideways rather than downwards, would fail here.
     */
    phase = 12;
    {
        sublet_cap fblk, fout, c1, ch1, c2, ch2, ob1, ob2;
        CHECK(ngx_subpool_block(4096, &fblk, &fout) == 1);

        sublet_carve(&fblk, sublet_base(&fblk) + 1024, &c1);
        sublet_handle(&c1, &ch1);
        sublet_carve(&c1, sublet_base(&c1) + 64, &ob1);
        unsigned char *p1 = (unsigned char *) sublet_take(&ob1);

        sublet_carve(&fblk, sublet_base(&fblk) + 1024, &c2);
        sublet_handle(&c2, &ch2);
        sublet_carve(&c2, sublet_base(&c2) + 64, &ob2);
        unsigned char *p2 = (unsigned char *) sublet_take(&ob2);

        CHECK(p1 != NULL && p2 != NULL && p1 != p2);
        if (p1 != NULL && p2 != NULL) {
            p1[0] = 0x11; p1[63] = 0x1F;
            p2[0] = 0x22; p2[63] = 0x2F;

            sublet_give_to(&ch1, &c1);        /* the first child revokes its own authority */

            CHECK(p2[0] == 0x22 && p2[63] == 0x2F);   /* the sibling still reads */
            p2[0] = 0x33; p2[63] = 0x3F;
            CHECK(p2[0] == 0x33 && p2[63] == 0x3F);   /* and still writes */

            /* And the first child's region came back to ITS bounds, not to the block's, which is
               what makes the revoke downward rather than sideways. */
            CHECK(sublet_end(&c1) - sublet_base(&c1) == 1024);
        }

        sublet_give_to(&fout, &fblk);         /* the ancestor takes the whole block back */
        CHECK(sublet_end(&fblk) - sublet_base(&fblk) == 4096);
        sublet_clear(&fblk);
    }

    phase = 13;
    size_t rest = ngx_subpool_arena_left;
    CHECK(rest > 0);
    sublet_cap rr, rh, r4, h4;
    CHECK(ngx_subpool_block(rest, &rr, &rh) == 1);
    CHECK(ngx_subpool_arena_left == 0);
    CHECK(ngx_subpool_block(16, &r4, &h4) == 0);

    /* The phase field carries WHERE it first failed when something did, and how far it got
       when nothing did. A count of failures without a place is not a result. */
    NGX_DOM_MARK((((failures ? failphase : phase) & 0xFF) << 16)
                 | ((ran & 0xFF) << 8) | (failures & 0xFF));
}
