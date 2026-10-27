/* Does the seam allocator behave, BEFORE any of PHP is pointed at it?
 *
 * WHY THIS COMES FIRST. Phase A's lesson, paid for twice: a probe that exercises the real code
 * in the shape the real caller uses it finds in one second what a 2.3 MB PHP image finds in
 * fifteen minutes, and names it. The identity widen passed a probe that tested the wrong half
 * of the problem; the widen that replaced it was caught by a stage that checked the cursor
 * rather than only the extent. So this drives zend-cap-blocks.c directly, through the same
 * hooks the patched zend_alloc.c calls, including the two shapes that actually bit:
 * AG(cache)'s hand-back, and the intrusive list walking a neighbour's header.
 *
 * It links the REAL zend-cap-blocks.c and zend-cap-backing.c. A probe with its own copy of the
 * logic would pass while the engine broke.
 *
 * ONE STAGE PER RUN, selected by argv: most stages are expected to FAULT and a capability fault
 * ends the process. Each prints what it is about to do, so the reading is "reached S, then
 * faulted" rather than an inference. Stage 0 is the positive control and must COMPLETE; if it
 * does not, nothing the others report means anything.
 *
 *   0  positive control: alloc, write, read back, retire, header still writable, free
 *   1  WRITE through a retired payload alias                        expect FAULT
 *   2  READ through a retired payload alias                         expect FAULT
 *   3  AG(cache)'s shape: retire, re-issue to a new tenant, then use the OLD alias   FAULT
 *   4  the intrusive list: three blocks, retire the middle, walk the neighbours' headers
 *   5  realloc moves the payload: contents preserved, the old alias dead              FAULT
 *   6  the CONTROL arm (revoke off): the same stale use must COMPLETE
 *   7  the bound: one byte past the request, exact vs stock rounding
 *   8  the PLATFORM column on a CACHE-ABSORBED free: must NOT fault
 *   9  the PLATFORM column on a REAL free: must fault
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
 */
#include "zend-cap-port.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REAL_SIZE(n) (((n) + 7UL) & ~7UL)
#define BLOCK_FOR(n) (ZEND_CAP_HDR + REAL_SIZE(n))

static void say(const char *s) { printf("%s\n", s); fflush(stdout); }

static void report(const char *what, void *c)
{
    printf("%s cursor=0x%lx bounds=[0x%lx,0x%lx) tag=%d\n", what,
           __builtin_capstone_cap_get_cursor(c), __builtin_capstone_cap_get_base(c),
           __builtin_capstone_cap_get_end(c), (int)__builtin_capstone_cap_get_tag(c));
    fflush(stdout);
}

/* What PHP's _emalloc does, in two lines: ask the seam for a block, then derive the payload. */
static void *emalloc_like(unsigned long n, void **header_out)
{
    void *hdr = zend_cap_block_alloc((size_t)BLOCK_FOR(n));
    if (!hdr) { say("PROBE FAIL: block_alloc returned NULL"); exit(2); }
    if (header_out) { *header_out = hdr; }
    return zend_cap_payload(hdr, (size_t)n);
}

/* What _efree does: recover the header by LOOKUP (never arithmetic), retire, and -- on the
 * path that does not park the block in the cache -- free it. */
static void efree_like(void *ptr, int really_free)
{
    void *hdr = zend_cap_header(ptr);
    zend_cap_retire(hdr);
    if (really_free) { zend_cap_block_free(hdr); }
}

int main(int argc, char **argv)
{
    int stage = (argc > 1) ? atoi(argv[1]) : 0;
    unsigned long stats[8];
    void *hdr = 0, *pay;

    printf("PROBE-SEAM stage=%d\n", stage);
    fflush(stdout);

    /* The arm under test is the full seam: exact bounds, per-object retire, and the loan
     * returned so mallocng revokes the block too. Stage 6 is the STOCK column (nothing revoked
     * anywhere, the loan leaked); stages 8 and 9 are the PLATFORM column (the loan returned,
     * no per-object retire), which is what makes a seam catch attributable to the seam. */
    zend_cap_set_mode(stage == 6 ? ZEND_CAP_MODE_EXACT
                    : (stage == 8 || stage == 9)
                        ? (ZEND_CAP_MODE_EXACT | ZEND_CAP_MODE_RETURN)
                        : (ZEND_CAP_MODE_EXACT | ZEND_CAP_MODE_REVOKE | ZEND_CAP_MODE_RETURN));

    if (stage == 7) {
        /* The bound, both ways, in one run: 9 bytes requested, so REAL_SIZE is 16 and index 9
         * is inside the rounding slack. With EXACT it must fault; this stage runs the ROUNDED
         * arm first (which must survive) and then re-runs exact, which must not. */
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
         * headers. Retiring a payload must leave every one of them usable. */
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
        say("S4 three blocks linked through their headers; retiring the middle payload");
        efree_like(p[1], 0);                       /* the AG(cache) park: retire, do not free */
        say("S4 unlinking the middle the way REMOVE_POINTER_FROM_LIST does");
        {
            struct hdrish *mid = (struct hdrish *)h[1];
            struct hdrish *last = (struct hdrish *)mid->pLast;
            struct hdrish *next = (struct hdrish *)mid->pNext;
            /* Both of these dereference a NEIGHBOUR's header through a capability stored in
             * the retired block's own header. Physically this is where a revoked block
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
        zend_cap_block_free(h[1]);
        say("S4 RESULT: PASS -- retiring a payload leaves every header in the list live, so\n"
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
        grown = zend_cap_block_realloc(hdr, (size_t)BLOCK_FOR(96));
        if (!grown) { say("S5 FAIL: realloc returned NULL"); return 1; }
        {
            void *newpay = zend_cap_payload(grown, 96);
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

    /* Stages 0, 1, 2, 3 and 6 share this opening. */
    pay = emalloc_like(64, &hdr);
    report("header  ", hdr);
    report("payload ", pay);
    memset(pay, 'q', 64);
    say("payload written");

    if (stage == 0) {
        if (((char *)pay)[63] != 'q') { say("S0 FAIL: readback wrong"); return 1; }
        say("S0 retiring the payload (the AG(cache) park: retire, do not free)");
        efree_like(pay, 0);
        /* PHP writes p->cached and reads p->size HERE, after the logical free. */
        {
            struct { void *a, *b; unsigned long s; } *h = (void *)hdr;
            h->s = 64;
            if (h->s != 64) { say("S0 FAIL: header not writable after the retire"); return 1; }
        }
        say("S0 header still readable and writable after the payload was retired");
        {
            void *again = zend_cap_payload(hdr, 64);     /* the cache hit */
            report("reissued", again);
            memset(again, 'r', 64);
            say("S0 re-issued payload written");
            efree_like(again, 1);
        }
        say("S0 RESULT: PASS -- split, retire, header use, re-issue and block free all behave");
        zend_cap_stats(stats);
        printf("S0 stats blocks=%lu frees=%lu issues=%lu retires=%lu revokes=%lu splits=%lu\n",
               stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]);
        return 0;
    }

    if (stage == 1) {
        efree_like(pay, 0);
        say("S1 payload retired; WRITING through the stale alias (expect a fault)");
        ((char *)pay)[0] = 'x';
        say("S1 RESULT: NO FAULT -- the stale payload alias is still writable. SEAM BROKEN.");
        return 1;
    }

    if (stage == 2) {
        unsigned long acc = 0, i;
        efree_like(pay, 0);
        say("S2 payload retired; READING through the stale alias (expect a fault)");
        for (i = 0; i < 16; i++) { acc += (unsigned char)((char *)pay)[i]; }
        printf("S2 RESULT: NO FAULT -- read 0x%lx through a retired alias. SEAM BROKEN.\n", acc);
        return 1;
    }

    if (stage == 3) {
        void *again;
        efree_like(pay, 0);
        again = zend_cap_payload(hdr, 64);
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
        say("S6 payload retired on the CONTROL arm (revoke off); writing through the alias");
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
         * 23 of the 31 cases are invisible to ASAN. With the loan returned but no per-object
         * retire, a use-after-free of a CACHED block must therefore survive. If this stage
         * faulted, the seam's catches could be the platform's and there would be no way to
         * tell them apart. */
        efree_like(pay, 0);                 /* the park: retire (a no-op here), do not free */
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
        efree_like(pay, 1);                 /* retire (no-op) AND return the loan */
        say("S9 real block free on the PLATFORM arm; using the alias (expect a FAULT)");
        ((char *)pay)[0] = 'x';
        say("S9 RESULT: NO FAULT -- returning the loan did not revoke the block. The platform's\n"
            "           own lifetime is not firing, so the three columns are not distinct.");
        return 1;
    }

    say("usage: probe-seam 0|1|2|3|4|5|6|7|8|9");
    return 2;
}
