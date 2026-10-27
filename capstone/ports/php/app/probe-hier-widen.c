/* Does the HIERARCHICAL bounds wrapper still need zend_cap_widen on the virtual profile?
 *
 * WHY THIS EXISTS. libc/php_capstone_zend_hier.c narrows every pointer it returns to the
 * caller's object ("narrow on the way up") and must therefore WIDEN any pointer that
 * re-enters the level below ("widen on the way down"). On the physical profile that widen is
 * mandatory and was found twice by faults: PHP's AG(cache) parks the pointer it is handed, and
 * the lower level's realloc copies a whole block's worth through whatever bound it is given.
 * zend_cap_widen recovers the block capability from the physical arena's slot table -- a table
 * the virtual profile does not have, because musl's mallocng is the lower level there.
 *
 * READING runtime/virtual/heap.c says the widen may be UNNECESSARY rather than
 * unimplementable: free() and realloc() take the caller's pointer only as an ADDRESS plus a
 * one-byte validity probe (`checked`, heap.c:203-210 -- `lbu zero, 0(p)` then
 * __capstone_malloc_find), and then operate through r->raw, the heap's OWN retained capability
 * over the block. If that reading is right, zend_cap_widen can be the identity on virtual and
 * the wrapper is reused unchanged. If it is wrong, Phase A needs a block registry.
 *
 * A READING IS NOT A RESULT, so this asks the machine. The geometry is PHP's exactly:
 * _emalloc does `p = malloc(48 + REAL_SIZE(n)); return p + 48`, the wrapper narrows that to
 * [block_base, object_cursor + n), and _efree recovers `ptr - 48` and frees THAT.
 *
 * ONE STAGE PER RUN, because stage 2 is expected to fault and a fault ends the process.
 *
 *   0  positive control: write the object, recover the header the way _efree does, free it
 *   1  realloc through a narrowed pointer, and check the payload survived the copy
 *   2  NEGATIVE CONTROL: one byte past the narrowed end must FAULT
 *   3  zend_cap_widen recovers the BLOCK extent, not merely something that works
 *   4  the AG(cache) hand-back: park a widened pointer, re-issue it to a larger tenant of the
 *      same size class, and write at the NEW size
 *
 * Stage 2 is what makes 0 and 1 mean anything. Without it "free survived a narrowed pointer"
 * is equally consistent with the narrowing never having happened -- the exact shape of clean
 * result this project keeps being bitten by.
 *
 * STAGES 3 AND 4 EXIST BECAUSE STAGES 0-2 PASSED AND THE CONCLUSION DRAWN FROM THEM WAS STILL
 * WRONG. They said free() and realloc() need no widen, which is true, and that was read as
 * "zend_cap_widen can be the identity", which is false: PHP's _efree does not just hand the
 * pointer down, it PARKS it in a size-CLASS cache that hands it back to a different tenant
 * (zend_alloc.c:272). A narrow capability parked there ends after the PREVIOUS tenant's
 * object, and the next one's own `p[length] = 0` faults at zend_alloc.c:404. Measured: with
 * the identity widen, all six spatial cases and the arm's own control faulted there during
 * zend_startup. Stage 4 reproduces that hand-back directly, so the next person to touch the
 * widen has a test that fails instead of an engine that will not start.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* php_virtual_lower.c -- the port's own widen, under test here. */
void *zend_cap_widen(void *p);
unsigned long zend_cap_block_end(void *p);

#define HDR 48UL            /* sizeof(zend_mem_header) on a 16-byte-pointer target */
#define OBJ 24UL            /* the caller's object; REAL_SIZE(24) == 24, so no slack */

static void say(const char *s) { printf("%s\n", s); fflush(stdout); }

/* zc_rebound (php_capstone_zend_hier.c:80): keep the BASE, tighten only the end. The base
 * must survive so every header in the chain stays reachable -- see that file. */
static void *rebound(void *p, unsigned long size)
{
    unsigned long b = __builtin_capstone_cap_get_base(p);
    unsigned long c = __builtin_capstone_cap_get_cursor(p);
    return __builtin_capstone_cap_shrink(p, b, c + size);
}

static void report(const char *what, void *p)
{
    printf("%s cursor=0x%lx bounds=[0x%lx,0x%lx) tag=%d\n", what,
           __builtin_capstone_cap_get_cursor(p),
           __builtin_capstone_cap_get_base(p),
           __builtin_capstone_cap_get_end(p),
           (int)__builtin_capstone_cap_get_tag(p));
    fflush(stdout);
}

int main(int argc, char **argv)
{
    int stage = (argc > 1) ? atoi(argv[1]) : 0;
    printf("PROBE-HIER stage=%d\n", stage);
    fflush(stdout);

    char *block = (char *)malloc(HDR + OBJ);
    if (!block) { say("PROBE FAIL: malloc returned NULL"); return 2; }
    report("block   ", block);

    /* What PHP hands its caller, and what the wrapper narrows it to. */
    char *object = block + HDR;
    char *narrow = (char *)rebound(object, OBJ);
    report("narrowed", narrow);
    if (__builtin_capstone_cap_get_end(narrow) != __builtin_capstone_cap_get_cursor(narrow) + OBJ) {
        say("PROBE FAIL: the narrow did not take -- every later reading would be vacuous");
        return 2;
    }

    memset(narrow, 0x5A, OBJ);
    say("object written through the narrowed capability");

    /* _efree's own arithmetic, on the narrowed pointer PHP parked and handed back. */
    char *header = narrow - HDR;
    report("header  ", header);

    if (stage == 0) {
        say("S0 freeing through the header recovered from the NARROWED object pointer");
        free(header);
        say("S0 RESULT: PASS -- free() needs no widen: it probes one byte at the cursor and\n"
            "           then works through the heap's own retained block capability, so the\n"
            "           identity widen is sound on this profile.");
        return 0;
    }

    if (stage == 1) {
        /* The physical port's _erealloc fault was the lower level copying a block's worth
         * through an object-sized bound. mallocng copies through r->raw instead. */
        say("S1 realloc(header, HDR + 2*OBJ) through the narrowed-derived pointer");
        char *moved = (char *)realloc(header, HDR + 2 * OBJ);
        if (!moved) { say("S1 FAIL: realloc returned NULL"); return 1; }
        report("moved   ", moved);
        unsigned long i, bad = 0;
        for (i = 0; i < OBJ; i++) {
            if ((unsigned char)moved[HDR + i] != 0x5Au) { bad = i + 1; break; }
        }
        if (bad) { printf("S1 FAIL: payload byte %lu did not survive the copy\n", bad - 1); return 1; }
        say("S1 RESULT: PASS -- realloc copied the whole block and the payload survived, so\n"
            "           it too works from the heap's own capability, not the caller's bound.");
        free(moved);
        return 0;
    }

    if (stage == 2) {
        say("S2 NEGATIVE CONTROL: writing one byte PAST the narrowed end (expect a fault)");
        narrow[OBJ] = 0x11;
        /* volatile read so the store cannot be dropped as dead. */
        printf("S2 RESULT: NO FAULT -- read back 0x%x. The bound is NOT being enforced, and\n"
               "           stages 0 and 1 therefore prove nothing.\n",
               (unsigned)(unsigned char)*(volatile char *)&narrow[OBJ]);
        return 1;
    }

    if (stage == 3) {
        /* Does the widen recover the BLOCK, or merely return something that does not fault?
         * Checked against the heap's own record of the block, not against a guess. */
        unsigned long widened_end, block_end = zend_cap_block_end(narrow);
        char *widened = (char *)zend_cap_widen(narrow);
        if (!block_end) { say("S3 FAIL: the heap does not know this block -- lookup is broken"); return 1; }
        widened_end = __builtin_capstone_cap_get_end(widened);
        report("widened ", widened);
        printf("S3 narrow end=0x%lx  widened end=0x%lx  block end=0x%lx\n",
               __builtin_capstone_cap_get_end(narrow), widened_end, block_end);
        if (__builtin_capstone_cap_get_cursor(widened) != __builtin_capstone_cap_get_cursor(narrow)) {
            say("S3 FAIL: the widen moved the cursor; _efree's ptr-48 would land elsewhere");
            return 1;
        }
        if (widened_end != block_end) {
            printf("S3 RESULT: FAIL -- the widen returned a capability ending at 0x%lx, not the\n"
                   "           block's 0x%lx. An identity widen reads exactly like this, and it\n"
                   "           poisons the next tenant of this size class.\n", widened_end, block_end);
            return 1;
        }
        say("S3 RESULT: PASS -- zend_cap_widen recovers the block extent with the cursor intact.");
        free(widened - HDR);
        return 0;
    }

    if (stage == 4) {
        /* PHP's cache hand-back, in miniature, with sizes that genuinely share a bucket.
         * CACHE_INDEX is REAL_SIZE(size)>>3 (zend_alloc.c:132,140), so 25 and 32 both land in
         * class 4 and the block _emalloc(25) requested -- 48 + REAL_SIZE(25) = 80 -- is handed
         * to a 32-byte tenant next. 24 and 25 would NOT share a class, which is why this stage
         * allocates its own block instead of reusing the one above. */
        const unsigned long A = 25UL, B = 32UL;     /* REAL_SIZE(A) == REAL_SIZE(B) == 32 */
        char *blk, *objA, *parked, *reissued;
        unsigned long i;

        blk = (char *)malloc(HDR + 32UL);
        if (!blk) { say("S4 FAIL: malloc"); return 2; }
        report("S4 block", blk);
        objA = (char *)rebound(blk + HDR, A);
        printf("S4 tenant A = %lu bytes, bounds end 0x%lx\n", A,
               __builtin_capstone_cap_get_end(objA));
        memset(objA, 'a', A);

        /* _efree's `p`: the pointer PHP parks in AG(cache), after the widen. */
        parked = (char *)zend_cap_widen(objA) - HDR;
        report("S4 parked", parked);

        /* _emalloc's cache hit: the parked block comes back and OUR wrapper re-bounds it to the
         * NEW caller's size. zc_rebound only ever shrinks, so if `parked` ended at A the new
         * tenant silently keeps A's bound -- which is the bug this stage exists for. */
        reissued = (char *)rebound(parked + HDR, B);
        report("S4 reissd", reissued);
        if (__builtin_capstone_cap_get_end(reissued)
            != __builtin_capstone_cap_get_cursor(reissued) + B) {
            printf("S4 RESULT: FAIL -- tenant B asked for %lu bytes and got a capability ending\n"
                   "           %lu bytes in: it inherited tenant A's bound from the cache.\n", B,
                   __builtin_capstone_cap_get_end(reissued)
                     - __builtin_capstone_cap_get_cursor(reissued));
            return 1;
        }
        printf("S4 tenant B = %lu bytes; writing its terminating NUL at index %lu "
               "(zend_alloc.c:404)\n", B, B - 1);
        reissued[B - 1] = 0;
        for (i = 0; i + 1 < B; i++) { reissued[i] = 'b'; }
        say("S4 RESULT: PASS -- a block parked by one tenant and re-issued to a larger tenant of\n"
            "           the same size class is writable to the NEW object's end. The identity\n"
            "           widen fails here, at the line PHP actually faulted on.");
        free(parked);
        return 0;
    }

    say("usage: probe-hier-widen 0|1|2|3|4");
    return 2;
}
