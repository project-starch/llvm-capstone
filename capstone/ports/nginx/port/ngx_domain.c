/* nginx's pool in a Capstone domain: does it run, and does what it hands out hold what was
 * written into it?
 *
 * This is the step before the discipline, and it is deliberately separate from it. If a scenario
 * fails here, it is the port; if one fails after ngx_palloc.c is patched, it is the discipline.
 * The MicroPython port learned that the expensive way and this one starts with it.
 *
 * Seven scenarios, each one something the pool promises:
 *
 *   1  a pool hands out small objects and what is written through them reads back
 *   2  asking for more than a block holds chains a second block, and the first objects survive it
 *   3  an allocation above the pool's max goes to the large path and is still readable
 *   4  ngx_pfree releases a large one, and the level below gets the bytes back
 *   5  ngx_reset_pool makes the pool usable again and the small objects are gone
 *   6  ngx_pcalloc returns zeroes, which is the one thing the pool promises about content
 *   7  a thousand create and destroy cycles leave the level below where they found it
 *
 * Scenario 7 is the one that would catch a leak in the port rather than in nginx: the pool's own
 * blocks come from the level below, so if the domain's arena shrinks over a thousand cycles, the
 * port lost blocks that upstream would have returned.
 *
 * WHAT IS NOT TESTED HERE is that a released object faults on the next touch. It does not yet:
 * nothing is revoked until the Sublet patch lands. Asking that question needs its own image
 * anyway, because a fault ends the domain and a domain that faulted cannot report the six results
 * beside it.
 */
#include <stddef.h>
#include <stdint.h>

#include "ngx_shim.h"

/* The prefix is ONE byte, not two, because the payload needs three: steps, checks and failures.
   With a two-byte prefix the step count lands in the same bits as the prefix and the OR eats it,
   which is what the first bisecting run reported and what cost a second look at a correct run. */
#define NGX_DOM_MARK(n) do { *res = 0x4E000000u | ((unsigned) (n) & 0x00FFFFFFu); return; } while (0)

/* The arena the host shares, and what the level below was told about it. The two arms differ
   here and only here: the unprotected one takes a plain region and pg_level0 carves ordinary
   pointers out of it, the Sublet one takes a LINEAR region so that what the pool carves can be
   revoked. Everything after this point, including all seven scenarios, is the same code. */
#ifdef NGX_SUBLET
#include "ngx_subpool.h"
static sublet_cap arena_slot;
#else
void pg_level0_init(void *region, size_t bytes);
static unsigned char *arena_base;
static size_t arena_bytes;
#endif

extern unsigned long ngx_level0_live;

/* ---- the scenarios ------------------------------------------------------- */

static unsigned failures;
static unsigned ran;

#define CHECK(cond) do { ++ran; if (!(cond)) { ++failures; } } while (0)

/* NGX_STOP_AFTER bisects the driver. A domain that faults returns nothing at all, so a run that
   only ever faults yields one bit per boot; stopping after step n and returning turns that into a
   number. Same reason MPY_STAGE exists in the MicroPython port, learned there the expensive way.
   Steps are counted by STEP() and the build passes the stop point. */
#ifndef NGX_STOP_AFTER
#define NGX_STOP_AFTER 0            /* 0 = run everything */
#endif
static unsigned step_no;
static int stopped;               /* a return from one scenario is not a stop of the driver, which
                                     is how the first bisection reached a path it meant to skip */
#define STEP() do { \
    ++step_no; \
    if (NGX_STOP_AFTER && step_no >= (unsigned) NGX_STOP_AFTER) { stopped = 1; return; } \
} while (0)

/* Written through an object and read back, so a pointer that merely looks right is not enough. */
static int write_read(unsigned char *p, size_t n, unsigned char seed) {
    for (size_t i = 0; i < n; i++) {
        p[i] = (unsigned char) (seed + i);
    }
    for (size_t i = 0; i < n; i++) {
        if (p[i] != (unsigned char) (seed + i)) {
            return 0;
        }
    }
    return 1;
}

static void scenario_small_and_blocks(void) {
    ngx_pool_t *pool = ngx_create_pool(1024, NULL);
    CHECK(pool != NULL);
    if (pool == NULL) {
        return;
    }

    STEP();                       /* 1: the pool was created */
    unsigned char *a = ngx_palloc(pool, 64);
    unsigned char *b = ngx_palloc(pool, 64);
    CHECK(a != NULL && b != NULL && a != b);
    CHECK(write_read(a, 64, 0x11));
    CHECK(write_read(b, 64, 0x22));
    STEP();                       /* 2: two small objects, written and read back */

    /* Past what one block holds, so the pool chains another. The first two objects must survive
       that, which is what a chain of blocks is for. */
    for (int i = 0; i < 64; i++) {
        unsigned char *c = ngx_palloc(pool, 64);
        CHECK(c != NULL);
        if (c == NULL) {
            break;
        }
    }
    STEP();                       /* 3: the block chain grew */
    CHECK(write_read(a, 64, 0x33));
    CHECK(write_read(b, 64, 0x44));

    /* Above the pool's max, so the large path and not the bump. */
    STEP();                       /* 4: the first objects survived the chain */
    unsigned char *big = ngx_palloc(pool, 8192);
    CHECK(big != NULL);
    if (big != NULL) {
        CHECK(write_read(big, 8192, 0x55));
        CHECK(ngx_pfree(pool, big) == NGX_OK);
    }

    /* Reset, then the pool works again. The small objects are gone, which is not checked by
       reading them: that would be a use after free, and this image cannot tell a trap from a
       crash. It is checked by the pool serving again from the start. */
    STEP();                       /* 5: the large path and its free */
    ngx_reset_pool(pool);
    unsigned char *d = ngx_palloc(pool, 64);
    CHECK(d != NULL);
    CHECK(write_read(d, 64, 0x66));

    STEP();                       /* 6: reset, and the pool serves again */
    unsigned char *z = ngx_pcalloc(pool, 256);
    CHECK(z != NULL);
    if (z != NULL) {
        int zeroed = 1;
        for (size_t i = 0; i < 256; i++) {
            if (z[i] != 0) {
                zeroed = 0;
            }
        }
        CHECK(zeroed);
    }

    STEP();                       /* 7: pcalloc zeroed */
    ngx_destroy_pool(pool);
    STEP();                       /* 8: destroy */
}

/* The three entry points the other two scenarios never call, and they are not obscure: nginx has
   328 calls to ngx_pnalloc, 46 to ngx_pool_cleanup_add, and two to ngx_pmemalign, both of which
   ask for an alignment far stricter than a capability. A port that answered only what its own
   driver happened to touch would be a port of six functions out of nine. */
static int cleanup_ran;

static void note_cleanup(void *data) {
    cleanup_ran = (int) (unsigned long) (*(unsigned char *) data);
}

static void scenario_api_surface(void) {
    ngx_pool_t *pool = ngx_create_pool(4096, NULL);
    CHECK(pool != NULL);
    if (pool == NULL) {
        return;
    }

    /* unaligned by contract, and the only thing promised is that it is usable */
    unsigned char *n1 = ngx_pnalloc(pool, 33);
    unsigned char *n2 = ngx_pnalloc(pool, 33);
    CHECK(n1 != NULL && n2 != NULL && n1 != n2);
    CHECK(write_read(n1, 33, 0x77));
    CHECK(write_read(n2, 33, 0x88));

    STEP();                       /* 9: pnalloc */

    /* A page, which is what ngx_radix_tree asks for. The address is only READ here. Casting it
       to an integer and back would hand over an address with no tag, which is cause 24 and a
       lesson this port has already paid for twice. */
    unsigned char *pg = ngx_pmemalign(pool, 4096, 4096);
    CHECK(pg != NULL);
    if (pg != NULL) {
        CHECK(((unsigned long) (void *) pg & 4095UL) == 0);
        CHECK(write_read(pg, 4096, 0x99));
    }

    /* And the direct IO case, a 512 byte boundary out of a block that starts wherever it starts */
    unsigned char *dio = ngx_pmemalign(pool, 1024, 512);
    CHECK(dio != NULL);
    if (dio != NULL) {
        CHECK(((unsigned long) (void *) dio & 511UL) == 0);
        CHECK(write_read(dio, 1024, 0xAA));
    }

    STEP();                       /* 10: pmemalign, at a page and at a sector */

    cleanup_ran = 0;
    ngx_pool_cleanup_t *c = ngx_pool_cleanup_add(pool, 1);
    CHECK(c != NULL);
    if (c != NULL) {
        *(unsigned char *) c->data = 0x5C;
        c->handler = note_cleanup;
    }

    ngx_destroy_pool(pool);
    CHECK(cleanup_ran == 0x5C);   /* the handler ran, and its data was still readable when it did */

    STEP();                       /* 11: a cleanup handler, run at destroy, reading pool memory */
}

/* A thousand cycles, and what nginx holds from the level below must end where it started. A
   pool's blocks come from there, so a port that lost one would show here and nowhere else. */
static void scenario_balance(void) {
    /* Whichever level this arm stands on. Under the discipline the pool never calls ngx_alloc, so
       ngx_level0_live would sit at zero and the check would pass without checking anything. */
#ifdef NGX_SUBLET
    unsigned long before = ngx_subpool_live;
#else
    unsigned long before = ngx_level0_live;
#endif
    for (int i = 0; i < 1000; i++) {
        ngx_pool_t *p = ngx_create_pool(1024, NULL);
        if (p == NULL) {
            ++failures;
            ++ran;
            return;
        }
        (void) ngx_palloc(p, 100);
        (void) ngx_palloc(p, 3000);      /* one small, one large */
        ngx_destroy_pool(p);
    }
    ++ran;
#ifdef NGX_SUBLET
    if (ngx_subpool_live != before) {
#else
    if (ngx_level0_live != before) {
#endif
        ++failures;
    }
}

/* ---- being a domain ------------------------------------------------------ */

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
        /* The arena. Its end comes from the base by pointer arithmetic and never from a cast of
           the builtin's integer: a cast carries the address and no tag, and everything carved
           from it afterwards would be untagged. The MicroPython port paid for that lesson. */
        unsigned long lo = (unsigned long) (void *) res;
        unsigned long hi = __builtin_capstone_cap_get_end((void *) res);
        arena_base = (unsigned char *) res;
        arena_bytes = (size_t) (hi - lo);
#endif
        return;
    }

    NGX_ANCHOR_RUNG();

#ifdef NGX_SUBLET
    /* A REV_SHARED share arrives NONLIN and csmrev would refuse it, three calls later and
       wearing a different face. Asked here instead. */
    if (sublet_type(&arena_slot) != 0) {
        NGX_DOM_MARK(0xFD0000u | (unsigned) sublet_type(&arena_slot));
    }
    ngx_subpool_init(&arena_slot);

#ifdef NGX_SUBLET_STAGE
    /* One pool API call per stage, each marking before the next runs, so a fault names the call
       that caused it rather than an address inside sublet.h. The MicroPython port's MPY_STAGE,
       for the same reason and with the same payoff. */
    {
        ngx_pool_t *sp; unsigned char *a, *b, *c; int st = NGX_SUBLET_STAGE;
        #define STAGE(n) do { if (st < (n)) { NGX_DOM_MARK(0xB00000u | (unsigned) st); } } while (0)
        STAGE(1);
        sp = ngx_create_pool(16384, NULL);
        STAGE(2);
        if (sp == NULL) { NGX_DOM_MARK(0xB1FFFFu); }
        a = ngx_palloc(sp, 64);  if (a == NULL) { NGX_DOM_MARK(0xB2FFFFu); }
        a[0] = 0x11; a[63] = 0x22;
        STAGE(3);
        b = ngx_palloc(sp, 4000); if (b == NULL) { NGX_DOM_MARK(0xB3FFFFu); }
        b[0] = 0x33;
        STAGE(4);
        b = ngx_palloc(sp, 8000); if (b == NULL) { NGX_DOM_MARK(0xB4FFFFu); }  /* forces a block */
        STAGE(5);
        c = ngx_palloc(sp, 100000); if (c == NULL) { NGX_DOM_MARK(0xB5FFFFu); } /* large path */
        c[0] = 0x44;
        STAGE(6);
        ngx_pfree(sp, c);
        STAGE(7);
        ngx_reset_pool(sp);
        STAGE(8);
        a = ngx_palloc(sp, 64); if (a == NULL) { NGX_DOM_MARK(0xB8FFFFu); }
        a[0] = 0x55;
        STAGE(9);
        ngx_destroy_pool(sp);
        NGX_DOM_MARK(0xBF0000u | (unsigned) st);
    }
#endif
#else
    if (arena_base == NULL) {
        NGX_DOM_MARK(0xFE);          /* no arena: the host never shared one */
    }

    pg_level0_init(arena_base, arena_bytes);
#endif
    failures = 0;
    ran = 0;

    scenario_small_and_blocks();
    if (!stopped) {
        scenario_api_surface();
    }
    if (!stopped) {
        scenario_balance();
    }

    /* checks run in bits 8..15, failures in bits 0..7, so a run that did nothing is not a pass */
    /* steps in bits 16..23 so a bisecting run says how far it got, checks in 8..15, failures in
       0..7, and a run that did nothing cannot read as a pass. */
    NGX_DOM_MARK(((step_no & 0xFF) << 16) | ((ran & 0xFF) << 8) | (failures & 0xFF));
}
