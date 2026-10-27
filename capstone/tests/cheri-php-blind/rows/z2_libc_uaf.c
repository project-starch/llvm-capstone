/* THE ABSOLUTE CONTROL: plain libc malloc -> free -> use. No PHP allocator, no
 * header, no size-class cache, nothing of ours between the defect and libc.
 *
 * Needed because every other temporal row MISSED under async revocation, and
 * a6 -- which does reach free() -- MISSED too. `malloc_revoke_enabled()`
 * returning 1 says the runtime has revocation TURNED ON; it does not say a
 * sweep ever RAN in a process this short-lived. Without this row, a MISS
 * everywhere cannot distinguish
 *
 *   (a) CHERI is blind to these defects because PHP's efree never calls free,
 *   from
 *   (b) revocation does not actually fire on this vehicle for this workload,
 *
 * and (a) is the claim being made, so (b) has to be excluded by measurement
 * rather than by the sysctl's say-so.
 *
 * Under revoke-on-every-free the dangling capability must be revoked here. If
 * even this MISSES, no MISS reported by any other row supports a claim about
 * CHERI -- it only says the sweep never ran -- and that must be stated plainly.
 */
#include "row.h"
#include <stdlib.h>
#include <string.h>

ROW_MAIN("z2_libc_uaf", {
    char *p = (char *) malloc(64);
    if (!p) { printf("MALLOC-FAILED\n"); }
    else {
        memset(p, 'L', 64);
        printf("TAG-BEFORE-FREE=%d\n", (int) __builtin_cheri_tag_get(p));
        free(p);
        /* Read the tag WITHOUT dereferencing: under revocation the capability is
         * untagged by the sweep, which is observable before any fault. */
        printf("TAG-AFTER-FREE=%d\n", (int) __builtin_cheri_tag_get(p));
        printf("LIBC-POST-FREE-READ byte0=%c\n", p[0]);   /* the actual UAF */
        row_consume(p, 64);
        printf("LIBC-UAF-SURVIVED\n");
    }
})
