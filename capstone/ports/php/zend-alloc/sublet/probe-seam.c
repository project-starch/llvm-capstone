/* Does the seam behave, BEFORE any of PHP is pointed at it?
 *
 * WHY THIS COMES FIRST. Phase A's lesson, paid for twice: a probe that exercises the real code
 * in the shape the real caller uses it finds in one second what a 2.3 MB PHP image finds in
 * fifteen minutes, and names it. So this drives zend-cap-mode.c directly, through the same
 * calls the patched zend_alloc.c's macros make, including the two shapes that actually bit:
 * AG(cache)'s hand-back, and the intrusive list walking a neighbour's header.
 *
 * It links the REAL zend-cap-mode.c. A probe with its own copy of the logic would pass while
 * the engine broke.
 *
 * ONE STAGE PER RUN, selected by argv: most stages are expected to FAULT and a capability fault
 * ends the process. Each prints what it is about to do, so the reading is "reached S, then
 * faulted" rather than an inference. Stage 0 is the positive control and must COMPLETE; if it
 * does not, nothing the others report means anything.
 *
 *   0  positive control: alloc, write, read back, reclaim, header still writable, free
 *   1  WRITE through a reclaimed payload alias                      expect FAULT
 *   2  READ through a reclaimed payload alias                       expect FAULT
 *   3  AG(cache)'s shape: reclaim, re-issue to a new tenant, then use the OLD alias   FAULT
 *   4  the intrusive list: three blocks, reclaim the middle, walk the neighbours' headers
 *   5  realloc moves the payload: contents preserved, the old alias dead              FAULT
 *   6  the CONTROL arm (revoke off): the same stale use must COMPLETE
 *   7  the bound: one byte past the request, exact vs stock rounding
 *   8  the PLATFORM column on a CACHE-ABSORBED free: must NOT fault
 *   9  the PLATFORM column on a REAL free: must fault
 *  10  a double efree: the second reclaim faults INSIDE the libc, at cap_malloc_reclaim
 *  11  a grandchild the caller derived for itself is refused by the reclaim (cause 29)
 *  12  an interior pointer handed to efree is refused by name (ZEND_CAP_FAIL code=4)
 *
 * Stage 6 is what makes 1, 2, 3 and 5 mean anything: it is the same access on an arm whose
 * only difference is that efree does not revoke. If stage 6 faulted too, the faults above
 * would not be about revocation at all.
 *
 * Stages 8 and 9 answer the question stage 6 cannot: whether a catch belongs to the SEAM or to
 * mallocng underneath it. The platform revokes a block when the block is freed, and PHP's
 * cache means most blocks never are -- so on the platform arm a cache-absorbed free must
 * survive (stage 8) while a real one must not (stage 9). Without that pair, every temporal
 * catch here would be unattributable between the two layers.
 *
 * Stages 10 and 11 are the identity checks the previous seam did in software (its issued flag
 * and its index) and that the libc's CREVOKE now does in hardware: the fault is in the
 * allocator, at a named site, which the judge attributes through the record's return address. */
#include "zend-cap-mode.h"
#include <malloc_capstone.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HDR 48UL   /* sizeof(zend_mem_header) on a 16-byte-pointer target; asserted by the engine */
#define REAL_SIZE(n) (((n) + 7UL) & ~7UL)
#define BLOCK_FOR(n) (HDR + REAL_SIZE(n))

static void say(const char *s) { printf("%s\n", s); fflush(stdout); }

static void report(const char *what, void *c)
{
    printf("%s cursor=0x%lx bounds=[0x%lx,0x%lx) tag=%d\n", what,
           __builtin_capstone_cap_get_cursor(c), __builtin_capstone_cap_get_base(c),
           __builtin_capstone_cap_get_end(c), (int)__builtin_capstone_cap_get_tag(c));
    fflush(stdout);
}

/* What PHP's _emalloc does: take a block, then issue the payload. */
static void *emalloc_like(unsigned long n, void **header_out)
{
    void *hdr = zend_cap_block((size_t)BLOCK_FOR(n));
    if (!hdr) { say("PROBE FAIL: block returned NULL"); exit(2); }
    if (header_out) { *header_out = hdr; }
    return zend_cap_issue(hdr, HDR, (size_t)n);
}

/* What _efree does: reclaim the block from the caller's pointer (which ends the caller's child),
 * then -- on the path that does not park the block in the cache -- release it. */
static void *efree_like(void *ptr, int really_free)
{
    void *hdr = zend_cap_reclaim(ptr, HDR);
    if (really_free) { zend_cap_release(hdr); }
    return hdr;
}

int main(int argc, char **argv)
{
    int stage = (argc > 1) ? atoi(argv[1]) : 0;
    unsigned long stats[8];
    void *hdr = 0, *pay;

    printf("PROBE-SEAM stage=%d\n", stage);
    fflush(stdout);

    /* The arm under test is the full seam: exact bounds, a child per object, and the block
     * returned so mallocng revokes it too. Stage 6 is the STOCK column (nothing revoked
     * anywhere, the block leaked); stages 8 and 9 are the PLATFORM column (the block returned,
     * no child per object), which is what makes a seam catch attributable to the seam. */
    zend_cap_set_mode(stage == 6 ? ZEND_CAP_MODE_EXACT
                    : (stage == 8 || stage == 9)
                        ? (ZEND_CAP_MODE_EXACT | ZEND_CAP_MODE_RETURN)
                        : (ZEND_CAP_MODE_EXACT | ZEND_CAP_MODE_REVOKE | ZEND_CAP_MODE_RETURN));

    if (stage == 7) {
        /* The bound, both ways, in one run: 9 bytes requested, so REAL_SIZE is 16 and index 9
         * is inside the rounding slack. With EXACT it must fault; this stage runs the ROUNDED
         * arm first (which must survive) and then re-runs exact, which must not. The two
         * report lines are what run-realphp.py's bound control reads, word for word. */
        volatile unsigned char seen;
        zend_cap_set_mode(ZEND_CAP_MODE_REVOKE | ZEND_CAP_MODE_RETURN);   /* rounded */
        pay = emalloc_like(9, &hdr);
        report("rounded ", pay);
        memset(pay, 'a', 9);
        seen = ((volatile unsigned char *)pay)[9];
        printf("S7 rounded arm read index 9 = 0x%02x -- the slack is reachable, as stock PHP\n",
               (unsigned)seen);
        fflush(stdout);
        efree_like(pay, 1);

        zend_cap_set_mode(ZEND_CAP_MODE_EXACT | ZEND_CAP_MODE_REVOKE
                          | ZEND_CAP_MODE_RETURN);
        pay = emalloc_like(9, &hdr);
        report("exact   ", pay);
        memset(pay, 'b', 9);
        say("S7 exact arm now reading index 9 (expect a fault)");
        seen = ((volatile unsigned char *)pay)[9];
        printf("S7 RESULT: NO FAULT -- read 0x%02x past a 9-byte request on the EXACT arm. "
               "The bound is not being applied.\n", (unsigned)seen);
        return 1;
    }

    if (stage == 4) {
        /* THE INTRUSIVE LIST, which is why the header must outlive the payload. ZEND_MM is
         * undefined in this build, so REMOVE_POINTER_FROM_LIST is real: it writes
         * p->pLast->pNext and p->pNext->pLast -- through capabilities stored in OTHER blocks'
         * headers. Reclaiming a payload must leave every one of them usable. */
        struct hdrish { void *pNext; void *pLast; unsigned long size_and_cached; };
        void *h[3];
        void *p[3];
        int i;
        for (i = 0; i < 3; i++) { p[i] = emalloc_like(40, &h[i]); memset(p[i], 'a' + i, 40); }
        /* Link them the way ADD_POINTER_TO_LIST would: h[2] -> h[1] -> h[0]. */
        ((struct hdrish *)h[0])->pNext = (void *)0;
        ((struct hdrish *)h[0])->pLast = h[1];
        ((struct hdrish *)h[1])->pNext = h[0];
        ((struct hdrish *)h[1])->pLast = h[2];
        ((struct hdrish *)h[2])->pNext = h[1];
        ((struct hdrish *)h[2])->pLast = (void *)0;
        say("S4 three blocks linked through their headers; reclaiming the middle payload");
        {
            /* The AG(cache) park: reclaim, do not free. What comes back is the block as the
             * heap holds it; PHP's own _efree continues on exactly that pointer. */
            struct hdrish *mid = (struct hdrish *)efree_like(p[1], 0);
            struct hdrish *last, *next;
            if ((void *)mid != h[1]) { say("S4 FAIL: the reclaim did not return the middle block"); return 1; }
            say("S4 unlinking the middle the way REMOVE_POINTER_FROM_LIST does");
            last = (struct hdrish *)mid->pLast;
            next = (struct hdrish *)mid->pNext;
            /* Both of these dereference a NEIGHBOUR's header through a capability stored in
             * the reclaimed block's own header. Physically this is where a revoked block
             * poisoned its neighbours and faulted with cause 24 inside the allocator. */
            if (last) { last->pNext = mid->pNext; }
            if (next) { next->pLast = mid->pLast; }
            printf("S4 unlink done: head's pLast now %p\n", next ? next->pLast : (void *)0);
            fflush(stdout);
        }
        /* And the other two payloads must still be readable. */
        if (((char *)p[0])[0] != 'a' || ((char *)p[2])[0] != 'c') {
            say("S4 FAIL: a neighbour's payload did not survive"); return 1;
        }
        efree_like(p[0], 1);
        efree_like(p[2], 1);
        zend_cap_release(h[1]);
        say("S4 RESULT: PASS -- reclaiming a payload leaves every header in the list live, so\n"
            "           PHP's own _efree ordering needs no change.");
        zend_cap_stats(stats);
        printf("S4 stats blocks=%lu frees=%lu issues=%lu retires=%lu revokes=%lu\n",
               stats[0], stats[1], stats[2], stats[3], stats[4]);
        return 0;
    }

    if (stage == 5) {
        void *grown;
        pay = emalloc_like(32, &hdr);
        memset(pay, 'z', 32);
        report("before  ", pay);
        say("S5 realloc to 96 bytes");
        /* What _erealloc does: recover the block WITHOUT ending anything, take a fresh block
         * with the old contents (the old block released, which ends the caller's child with
         * it), issue the new payload. */
        hdr = zend_cap_header(pay, HDR);
        grown = zend_cap_reblock(hdr, (size_t)BLOCK_FOR(96), HDR + 32);
        if (!grown) { say("S5 FAIL: reblock returned NULL"); return 1; }
        {
            void *newpay = zend_cap_issue(grown, HDR, 96);
            unsigned long i;
            report("after   ", newpay);
            for (i = 0; i < 32; i++) {
                if (((char *)newpay)[i] != 'z') {
                    printf("S5 FAIL: byte %lu did not survive the copy\n", i); return 1;
                }
            }
            say("S5 contents survived the move; now using the OLD payload alias (expect a fault)");
            ((char *)pay)[0] = 'x';
            say("S5 RESULT: NO FAULT -- the old payload alias outlived the realloc.");
            return 1;
        }
    }

    /* Stages 0, 1, 2, 3, 6, 8, 9, 10 and 11 share this opening. */
    pay = emalloc_like(64, &hdr);
    report("header  ", hdr);
    report("payload ", pay);
    memset(pay, 'q', 64);
    say("payload written");

    if (stage == 0) {
        void *back;
        if (((char *)pay)[63] != 'q') { say("S0 FAIL: readback wrong"); return 1; }
        say("S0 reclaiming the payload (the AG(cache) park: reclaim, do not free)");
        back = efree_like(pay, 0);
        if (back != hdr) { say("S0 FAIL: the reclaim did not return the block"); return 1; }
        /* PHP writes p->cached and reads p->size HERE, after the logical free. */
        {
            struct { void *a, *b; unsigned long s; } *h = (void *)back;
            h->s = 64;
            if (h->s != 64) { say("S0 FAIL: header not writable after the reclaim"); return 1; }
        }
        say("S0 header still readable and writable after the payload was reclaimed");
        {
            void *again = zend_cap_issue(back, HDR, 64);     /* the cache hit */
            report("reissued", again);
            memset(again, 'r', 64);
            say("S0 re-issued payload written");
            efree_like(again, 1);
        }
        say("S0 RESULT: PASS -- issue, reclaim, header use, re-issue and block free all behave");
        zend_cap_stats(stats);
        printf("S0 stats blocks=%lu frees=%lu issues=%lu retires=%lu revokes=%lu splits=%lu\n",
               stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]);
        return 0;
    }

    if (stage == 1) {
        efree_like(pay, 0);
        say("S1 payload reclaimed; WRITING through the stale alias (expect a fault)");
        ((char *)pay)[0] = 'x';
        say("S1 RESULT: NO FAULT -- the stale payload alias is still writable. SEAM BROKEN.");
        return 1;
    }

    if (stage == 2) {
        unsigned long acc = 0, i;
        efree_like(pay, 0);
        say("S2 payload reclaimed; READING through the stale alias (expect a fault)");
        for (i = 0; i < 16; i++) { acc += (unsigned char)((char *)pay)[i]; }
        printf("S2 RESULT: NO FAULT -- read 0x%lx through a reclaimed alias. SEAM BROKEN.\n", acc);
        return 1;
    }

    if (stage == 3) {
        void *again;
        void *back = efree_like(pay, 0);
        again = zend_cap_issue(back, HDR, 64);
        memset(again, 's', 64);
        printf("S3 region re-issued as %p and written; the old alias %p must be dead\n",
               again, pay);
        say("S3 now writing through the OLD alias (expect a fault)");
        ((char *)pay)[0] = 'x';
        say("S3 RESULT: NO FAULT -- the cache hand-back resurrected the old alias. SEAM BROKEN.");
        return 1;
    }

    if (stage == 6) {
        /* The control arm: identical code, revocation off. */
        efree_like(pay, 0);
        say("S6 payload reclaimed on the CONTROL arm (revoke off); writing through the alias");
        ((char *)pay)[0] = 'x';
        if (((char *)pay)[0] != 'x') { say("S6 FAIL: write did not land"); return 1; }
        say("S6 RESULT: PASS -- with revocation off the stale alias is still usable, so the\n"
            "           faults on the other stages are about revocation and nothing else.");
        zend_cap_stats(stats);
        printf("S6 stats retires=%lu revokes=%lu (both must be 0 on this arm)\n",
               stats[3], stats[4]);
        return (stats[3] || stats[4]) ? 1 : 0;
    }

    if (stage == 8) {
        /* THE PLATFORM COLUMN, on a block AG(cache) absorbed. mallocng revokes a block when it
         * is FREED, and PHP's cache means most blocks are never freed -- which is precisely why
         * so many of the corpus's use-after-frees are invisible to ASAN. With the block returned
         * but no child per object, a use-after-free of a CACHED block must therefore survive. If
         * this stage faulted, the seam's catches could be the platform's and there would be no
         * way to tell them apart. */
        efree_like(pay, 0);                 /* the park: reclaim (a no-op here), do not free */
        say("S8 cached-block free on the PLATFORM arm; using the alias (expect NO fault)");
        ((char *)pay)[0] = 'x';
        if (((char *)pay)[0] != 'x') { say("S8 FAIL: the write did not land"); return 1; }
        say("S8 RESULT: PASS -- the platform alone does NOT see a free that AG(cache) absorbed,\n"
            "           so a catch on the seam arm is the seam's and not mallocng's.");
        return 0;
    }

    if (stage == 9) {
        /* The same arm, on a block that really is freed: here the platform DOES catch it, and
         * saying so is what keeps stage 8 honest rather than a claim that mallocng does
         * nothing. */
        efree_like(pay, 1);                 /* reclaim (no-op) AND return the block */
        say("S9 real block free on the PLATFORM arm; using the alias (expect a FAULT)");
        ((char *)pay)[0] = 'x';
        say("S9 RESULT: NO FAULT -- returning the block did not revoke it. The platform's\n"
            "           own lifetime is not firing, so the three columns are not distinct.");
        return 1;
    }

    if (stage == 10) {
        efree_like(pay, 0);
        say("S10 payload reclaimed once; reclaiming it AGAIN (expect a fault at cap_malloc_reclaim)");
        efree_like(pay, 0);
        say("S10 RESULT: NO FAULT -- a double efree was not refused. IDENTITY CHECK BROKEN.");
        return 1;
    }

    if (stage == 11) {
        void *grand = __capstone_malloc_derive(pay, 0, 16);   /* what a client could do itself */
        if (!grand) { say("S11 FAIL: the grandchild derive returned NULL"); return 1; }
        say("S11 handing a GRANDCHILD to efree (expect a fault at cap_malloc_reclaim, cause 29)");
        efree_like(grand, 0);
        say("S11 RESULT: NO FAULT -- a grandchild was accepted as the issued child. IDENTITY CHECK BROKEN.");
        return 1;
    }

    if (stage == 12) {
        say("S12 handing an INTERIOR pointer (payload + 1) to efree (expect ZEND_CAP_FAIL code=4, exit 134)");
        efree_like((char *)pay + 1, 0);
        say("S12 RESULT: NO REFUSAL -- an interior pointer was accepted as the issued payload.");
        return 1;
    }

    say("usage: probe-seam 0|1|2|3|4|5|6|7|8|9|10|11|12");
    return 2;
}
