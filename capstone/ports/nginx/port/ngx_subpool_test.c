/* Does the level below keep its promise? A domain that asks the hardware, before nginx is patched.
 *
 * The claim this port makes is that a pool's whole contents die in one revocation.
 * port/ngx_subpool.c is the piece that has to make that true, and it is the piece to test first,
 * because a fault inside a patched ngx_palloc.c would be much harder to read back to its cause.
 * The PostgreSQL port tests its own level below the same way and for the same reason.
 *
 * Eight scenarios, each a claim the design makes:
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
 *   8  an arena spent to the LAST BYTE still refuses the next request. Not the same claim as 6:
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

static unsigned ran, failures, phase;
#define CHECK(cond) do { ++ran; if (!(cond)) { ++failures; } } while (0)

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

void domain_main(unsigned *res, unsigned func) {
    if (func == 1) {
        /* The arena, and it must arrive LINEAR or nothing below can ever be revoked. The type is
           checked rather than assumed: a REV_SHARED share arrives NONLIN, csmrev refuses it, and
           the failure would otherwise show up three calls later as something else. */
        sublet_store(&arena_slot, res);
        return;
    }

    if (sublet_type(&arena_slot) != 0) {         /* 0 = CAP_TYPE_LIN */
        NGX_DOM_MARK(0xFD0000u | (unsigned) sublet_type(&arena_slot));
    }

    ngx_subpool_init(&arena_slot);
    ran = failures = phase = 0;

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
    phase = 8;
    size_t rest = ngx_subpool_arena_left;
    CHECK(rest > 0);
    sublet_cap rr, rh, r4, h4;
    CHECK(ngx_subpool_block(rest, &rr, &rh) == 1);
    CHECK(ngx_subpool_arena_left == 0);
    CHECK(ngx_subpool_block(16, &r4, &h4) == 0);

    NGX_DOM_MARK(((phase & 0xFF) << 16) | ((ran & 0xFF) << 8) | (failures & 0xFF));
}
