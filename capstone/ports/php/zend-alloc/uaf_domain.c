/* Temporal axis: revoke-on-free in PHP 5.0.0's Zend allocator.
 *
 * SYNTHETIC, and labelled as such. This is NOT a CRASH-nnn corpus case. All 23
 * cache-masked use-after-free cases in the corpus are entangled with the
 * zval/refcount/executor machinery, so none is portable at this size. What this
 * reproduces is the allocator-level property the corpus itself isolates in
 * logic-bugs.md:643-644:
 *
 *     efree() deposits into the cache, then write through the stale pointer -> ASan SILENT
 *     efree() calls real free(), same write                                 -> heap-use-after-free
 *
 * ASan is silent in the first row because the block is never released: it is
 * parked in AG(cache) (zend_alloc.c:271-278) and the shadow still says
 * "addressable". The pointer is dangling by PHP's own contract and perfectly
 * legal to ASan.
 *
 * Under revoke-on-free the stale alias is dead the moment _efree runs, whether
 * or not the block is recycled -- no shadow memory, no quarantine, no redzone.
 *
 * ARMS
 *   fault    -DZEND_TEMPORAL                  -> halts on the stale write
 *   control  -DZEND_TEMPORAL -DZEND_NO_REVOKE -> completes, *res = ZEND_UAF_SURVIVED
 *
 * The control removes exactly one instruction, the revoke. It MUST complete: at
 * -O0 a plain spill/reload can produce a tag-gone fault that looks identical to
 * a caught use-after-free (ALLOCATOR-CONTRACT.md §4).
 *
 * KEEPING THE TWO AXES APART. The buffer is correctly sized and every access is
 * well inside its bounds, so a spatial fault cannot masquerade as a temporal
 * one. The expected diagnostic is "Cap mem access requires capability" (tag
 * gone, cause 24) or "on revoked capability" (cause 25) -- NOT the
 * "Cap mem access OOB" line the CRASH-008 arm produces.
 *
 * -O0 IS REQUIRED. At -O1+ the stale store can be hoisted above the free or
 * elided outright, and the access under test is never emitted.
 */
#include "zend_capstone_alloc.h"

#define ZEND_UAF_SURVIVED  0xD5U
#define ZEND_UAF_SETUP     0xBADU
#define ZEND_UAF_NOREUSE   0xBAEU   /* the block was NOT recycled -> test is vacuous */

void domain_main(unsigned *res, unsigned func)
{
    (void)func;

    /* 32 bytes: cache_index = REAL_SIZE(32)>>3 = 4, inside MAX_CACHED_MEMORY (11),
     * so PHP parks this in AG(cache) and hands it straight back next time. */
    char *p = (char *) _emalloc(32);
    if (!p) { *res = ZEND_UAF_SETUP; return; }
    for (int i = 0; i < 32; i++) { p[i] = (char)('A' + (i & 15)); }

    volatile char live = p[0];          /* in-bounds, pre-free: must work */
    if (live != 'A') { *res = ZEND_UAF_SETUP; return; }

    unsigned long was = __builtin_capstone_cap_get_base(p);

    _efree(p);                          /* parks in AG(cache); revokes under ZEND_TEMPORAL */

    /* Reallocate the same size class. This is what makes the case real rather
     * than academic: PHP hands back the SAME block, so on stock PHP the stale
     * `p` now aliases `q`'s memory -- the classic reuse hazard, and invisible to
     * ASan because the block was never released. */
    char *q = (char *) _emalloc(32);
    if (!q) { *res = ZEND_UAF_SETUP; return; }

    /* POSITIVE CONTROL ON THE INSTRUMENT. If the allocator did not actually
     * recycle the block, `p` and `q` are unrelated and a fault (or its absence)
     * would say nothing about use-after-free. Fail loudly rather than pass
     * quietly. */
    if (__builtin_capstone_cap_get_base(q) != was) { *res = ZEND_UAF_NOREUSE; return; }

    q[0] = 'n';                         /* the new tenant: must work */

    /* THE DEFECT. `p` is the caller's stale copy of a block now owned by `q`.
     * Well inside the original 32 bytes, so this is a temporal violation only. */
    p[1] = 'Z';

    volatile char readback = q[1];      /* on stock PHP this reads 'Z': p corrupted q */
    (void)readback;

    *res = ZEND_UAF_SURVIVED;           /* reached only when the stale access returned */
}
