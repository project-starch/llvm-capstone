/* Rounding-hidden over-read: the shape of the corpus's 6 CWE-125 cases.
 * emalloc(26) allocates REAL_SIZE(26)=32 bytes of payload under the real
 * allocator, so bytes 26..31 are slack INSIDE the malloc'd block. malloc bounds
 * the block and (char*)p+hdr inherits those bounds, so reading the slack is
 * in-bounds and tagged.
 *
 * The read offsets come from STOCK_REAL_SIZE, not from the live REAL_SIZE, so
 * this arm and the -DPHP_PATCHED_ALLOC arm perform the IDENTICAL access and
 * differ only in how much the allocator reserved for it. */
#include "row.h"
#include <string.h>
ROW_MAIN("b1_oob_read_slack", {
    const size_t n = 26;
    const size_t slack_end = STOCK_REAL_SIZE(26);       /* 32, both arms */
    char *s = (char *) emalloc(n);
    memset(s, 'x', n);
    printf("SLACK n=%zu this_arm_real=%zu reading [%zu..%zu]\n",
           n, php_alloc_real_size(n), n, slack_end - 1);
    size_t i;
    for (i = n; i < slack_end; i++) {
        printf("  s[%zu]=%02x\n", i, (unsigned char) s[i]);  /* over-read */
    }
    row_consume(s + n, slack_end - n);
    efree(s);
})
