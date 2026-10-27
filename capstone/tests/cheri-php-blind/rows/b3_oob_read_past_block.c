/* POSITIVE CONTROL, SPATIAL. One byte further than b1: past STOCK_REAL_SIZE, so
 * past the malloc'd block in BOTH arms. CHERI must fault here in every config
 * and both arms. If it does not, bounds are not being enforced on this vehicle
 * and no MISS reported by any other row means anything. */
#include "row.h"
#include <string.h>
ROW_MAIN("b3_oob_read_past_block", {
    const size_t n = 26;
    const size_t past = STOCK_REAL_SIZE(26);            /* 32, both arms */
    char *s = (char *) emalloc(n);
    memset(s, 'x', n);
    printf("PAST-BLOCK n=%zu this_arm_real=%zu reading s[%zu]\n",
           n, php_alloc_real_size(n), past);
    printf("  s[%zu]=%02x\n", past, (unsigned char) s[past]);   /* MUST FAULT */
    row_consume(s + past, 1);
    printf("NO-FAULT-PAST-BLOCK\n");
    efree(s);
})
