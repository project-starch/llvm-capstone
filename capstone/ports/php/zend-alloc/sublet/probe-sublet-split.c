/* Does the PHP seam design actually work on the virtual heap?
 *
 * WHY THIS EXISTS, AND WHY IT COMES BEFORE ANY PHP CODE. The retarget plan's allocator seam
 * rests on one claim: that a block lent by the virtual heap can be SPLIT into a header region
 * and a payload region, so that retiring the payload leaves the header live. PHP needs exactly
 * that, because zend_alloc.c reads p->size and writes p->cached AFTER the logical free, and
 * parks the block in AG(cache) with its pNext/pLast still pointing at neighbours. If the split
 * cannot separate the two, the whole design collapses back onto the physical port's
 * unlink-before-revoke deviation -- and we would rather learn that from 200 lines than from a
 * 2.3 MB PHP image whose fault lands somewhere in Zend.
 *
 * It also settles the two open risks the plan names:
 *   - stage 4: does reading the CURSOR of a revoked capability trap? ZEND_CAP_HEADER(ptr) would
 *     do that on a double-efree, and a fault inside the allocator is refused by the harness as a
 *     port artefact, so the fault has to land at a named probe label instead.
 *   - stage 5: must a split land on a 16-byte grain? sublet.h says Sublet regions are "aligned
 *     and sized to whole capabilities (16 bytes)"; PHP's header is 48 bytes, which is a grain
 *     multiple, but nothing in PHP guarantees that and the answer decides whether the adapter
 *     must pad. Stage 5 tests the SPLIT only, not a revoke of the resulting region -- see its
 *     own result text.
 *
 * ONE STAGE PER RUN, SELECTED BY ARGV. Most stages are expected to FAULT, and a fault ends the
 * process -- so stages cannot share a run. Each prints a line before the operation under test,
 * so the harness reads "reached S, then faulted" rather than having to infer it. Stage 0 is the
 * positive control: it must COMPLETE, and if it does not, nothing the other stages report means
 * anything.
 */
#include <sublet/sublet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned long __capstone_sublet_malloc_linear(size_t, sublet_cap *);
void          __capstone_sublet_free_linear(unsigned long);

#define HDR   48UL          /* sizeof(zend_mem_header) on a 16-byte-pointer target */
#define BLOCK 256UL         /* header + a 208-byte payload; both ends 16-aligned    */

/* Slots are PLACES: a capability an allocator owns lives in memory, never in a C variable
 * (sublet.h). These are file-scope for that reason, not for convenience. */
static sublet_cap block;    /* the loan, then the header region, then the header handle */
static sublet_cap senior;   /* senior to everything carved out of the loan              */
static sublet_cap payload;  /* the payload region, then its handle                      */

static void say(const char *s) { printf("%s\n", s); fflush(stdout); }

/* Lend a block and split it the way the PHP adapter would. Returns the block base. */
static unsigned long carve(void **hdr_out, void **pay_out)
{
    unsigned long base = __capstone_sublet_malloc_linear(BLOCK, &block);
    if (!base) { say("PROBE FAIL: the heap would not lend a block"); exit(2); }
    printf("loan base=0x%lx type=%lu bounds=[0x%lx,0x%lx)\n",
           base, sublet_type(&block), sublet_base(&block), sublet_end(&block));
    fflush(stdout);

    /* Senior handle BEFORE any split, so one revoke can reclaim the whole block later. */
    sublet_handle(&block, &senior);
    /* [block] = [base, base+HDR) ; [payload] = [base+HDR, base+BLOCK) */
    sublet_split(&block, base + HDR, &payload);

    *hdr_out = sublet_take(&block);
    *pay_out = sublet_take(&payload);
    return base;
}

int main(int argc, char **argv)
{
    int stage = (argc > 1) ? atoi(argv[1]) : 0;
    void *hdr = 0, *pay = 0;
    unsigned long base;

    printf("PROBE stage=%d\n", stage);
    fflush(stdout);

    if (stage == 5) {
        /* Does a split have to land on a 16-byte grain? Ask before assuming. */
        base = __capstone_sublet_malloc_linear(BLOCK, &block);
        if (!base) { say("PROBE FAIL: no loan"); return 2; }
        sublet_handle(&block, &senior);
        say("S5 splitting at base+40, which is NOT a 16-byte multiple");
        sublet_split(&block, base + 40, &payload);
        printf("S5 survived: lo=[0x%lx,0x%lx) hi=[0x%lx,0x%lx)\n",
               sublet_base(&block), sublet_end(&block),
               sublet_base(&payload), sublet_end(&payload));
        say("S5 RESULT: sublet_split at a non-16-multiple is PERMITTED and the bounds are\n"
            "           exact. NOTE THE LIMIT OF THIS READING: it says nothing about\n"
            "           revoking such a region -- sublet_give_to fills in 16-byte steps,\n"
            "           so a non-grain region is still untested there. PHP does not need\n"
            "           it: sizeof(zend_mem_header) is 48, a grain multiple, so the\n"
            "           adapter never splits off-grain and needs no padding.");
        return 0;
    }

    base = carve(&hdr, &pay);
    printf("hdr=%p pay=%p\n", hdr, pay);
    fflush(stdout);

    /* Both aliases must work before any of the retire stages mean anything. */
    memset(hdr, 0xA5, HDR);
    memset(pay, 0x5A, BLOCK - HDR);
    say("both aliases written");

    if (stage == 0) {
        /* THE POSITIVE CONTROL, and the design claim in miniature. */
        sublet_give(&payload);
        say("S0 payload retired");
        /* PHP writes p->cached and reads p->size here, through the header. */
        memset(hdr, 0x3C, HDR);
        if (((unsigned char *)hdr)[0] != 0x3C) { say("S0 FAIL: header readback wrong"); return 1; }
        say("S0 header still writable after the payload was retired");

        void *pay2 = sublet_take(&payload);
        memset(pay2, 0x77, BLOCK - HDR);
        printf("S0 re-issued payload alias=%p, written\n", pay2);
        sublet_give(&payload);
        /* One revoke on the senior handle ends every descendant. */
        sublet_give_to(&senior, &block);
        say("S0 senior revoke done");
        __capstone_sublet_free_linear(base);
        say("S0 RESULT: PASS -- split, retire, re-issue and senior revoke all behave");
        return 0;
    }

    if (stage == 1) {
        sublet_give(&payload);
        say("S1 payload retired; now WRITING through the stale alias (expect a fault)");
        memset(pay, 0x11, 16);
        say("S1 RESULT: NO FAULT -- the stale payload alias is still usable. DESIGN BROKEN.");
        return 1;
    }

    if (stage == 2) {
        sublet_give(&payload);
        say("S2 payload retired; now READING through the stale alias (expect a fault)");
        unsigned long acc = 0;
        for (unsigned i = 0; i < 16; i++) acc += ((unsigned char *)pay)[i];
        printf("S2 RESULT: NO FAULT -- read 0x%lx through a retired alias. DESIGN BROKEN.\n", acc);
        return 1;
    }

    if (stage == 3) {
        /* The AG(cache) case: the region is re-issued to a new tenant, and the PREVIOUS
         * tenant's alias must stay dead even though the bytes are live again. */
        sublet_give(&payload);
        void *pay2 = sublet_take(&payload);
        memset(pay2, 0x22, 16);
        printf("S3 region re-issued as %p and written; old alias %p must be dead\n", pay2, pay);
        say("S3 now writing through the OLD alias (expect a fault)");
        memset(pay, 0x33, 16);
        say("S3 RESULT: NO FAULT -- re-issue resurrected the old alias. DESIGN BROKEN.");
        return 1;
    }

    if (stage == 4) {
        /* Does metadata-only inspection of a dead capability trap? ZEND_CAP_HEADER would. */
        sublet_give(&payload);
        say("S4 payload retired; now reading the CURSOR of the stale alias, no dereference");
        unsigned long c = __builtin_capstone_cap_get_cursor(pay);
        printf("S4 RESULT: cursor read SURVIVED (0x%lx) -- a dead pointer can be mapped to its\n", c);
        say("           record without trapping, so the adapter must probe explicitly.");
        return 0;
    }

    say("usage: probe-sublet-split 0|1|2|3|4|5");
    return 2;
}
