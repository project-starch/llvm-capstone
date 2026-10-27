/* Cache-hidden use-after-free, read. The shape of most of the corpus's 23
 * CWE-416 cases: a buffer is efree'd and then read through the stale pointer.
 * Stock allocator: efree pushes the block into AG(cache)[4] and never calls
 * free(), so the capability stays tagged and in bounds. */
#include "row.h"
#include <string.h>
ROW_MAIN("a1_uaf_read", {
    char *s = (char *) emalloc(32);
    memcpy(s, "the quick brown fox jumped ok!!", 31);
    s[31] = 0;
    efree(s);
    printf("STALE-READ [%s]\n", s);   /* use-after-free */
    row_consume(s, 32);
})
