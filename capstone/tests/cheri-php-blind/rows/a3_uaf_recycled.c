/* Stale-but-reallocated: the 1-byte -> 8-byte same-bucket recycle. REAL_SIZE(1)
 * and REAL_SIZE(8) are both 8, so cache_index is 1 for both and the second
 * emalloc re-serves the very block the first one freed. The old owner then
 * reads what the new owner wrote. No free() is involved at any point, so this
 * is unreachable for ANY revocation policy -- the PHP analogue of the SQLite
 * baseline's row3_reuse headline. */
#include "row.h"
#include <string.h>
ROW_MAIN("a3_uaf_recycled", {
    char *old = (char *) emalloc(1);
    old[0] = 'A';
    efree(old);
    char *new = (char *) emalloc(8);   /* same bucket -> same block */
    memcpy(new, "SECRET7", 8);
    printf("SAME-BLOCK=%d\n", (old == new));
    printf("STALE-SEES [%c] via the freed pointer\n", old[0]);  /* reads 'S' */
    row_consume(old, 8);
    efree(new);
})
