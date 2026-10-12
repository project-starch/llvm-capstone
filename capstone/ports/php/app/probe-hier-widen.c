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
 *
 * Stage 2 is what makes 0 and 1 mean anything. Without it "free survived a narrowed pointer"
 * is equally consistent with the narrowing never having happened -- the exact shape of clean
 * result this project keeps being bitten by.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    say("usage: probe-hier-widen 0|1|2");
    return 2;
}
