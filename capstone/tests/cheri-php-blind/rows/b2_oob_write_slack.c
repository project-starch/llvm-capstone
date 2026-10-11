/* Rounding-hidden over-WRITE: the corpus's CWE-787 case (CRASH-008). Same
 * geometry as b1 but a store, which is the class capability hardware is
 * normally strongest against -- and still in bounds here. Offsets fixed from
 * STOCK_REAL_SIZE so both arms write the same bytes. */
#include "row.h"
#include <string.h>
ROW_MAIN("b2_oob_write_slack", {
    const size_t n = 26;
    const size_t slack_end = STOCK_REAL_SIZE(26);       /* 32, both arms */
    char *s = (char *) emalloc(n);
    memset(s, 'x', n);
    printf("SLACK n=%zu this_arm_real=%zu writing [%zu..%zu]\n",
           n, php_alloc_real_size(n), n, slack_end - 1);
    size_t i;
    for (i = n; i < slack_end; i++) { s[i] = (char) ('0' + (i & 7)); }  /* over-write */
    row_consume(s, slack_end);
    printf("OVERWRITE-RETURNED\n");
    efree(s);
})
