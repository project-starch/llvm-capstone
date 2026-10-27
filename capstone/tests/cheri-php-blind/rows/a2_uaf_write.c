/* Cache-hidden use-after-free, write. */
#include "row.h"
#include <string.h>
ROW_MAIN("a2_uaf_write", {
    char *s = (char *) emalloc(32);
    memset(s, 'a', 32);
    efree(s);
    memset(s, 'Z', 32);               /* use-after-free, store */
    printf("STALE-WRITE byte0=%c\n", s[0]);
    row_consume(s, 32);
})
