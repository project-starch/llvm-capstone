/* VEHICLE SANITY: the harness itself must run clean purecap in both arms and
 * all three configs. A fault here invalidates every other row. */
#include "row.h"
#include <string.h>
ROW_MAIN("z0_sanity", {
    int i;
    char *keep[8];
    for (i = 0; i < 8; i++) {
        keep[i] = (char *) emalloc(8 * (i + 1) + 3);
        memset(keep[i], 'k', 8 * (i + 1) + 3);
    }
    for (i = 0; i < 8; i++) { row_consume(keep[i], 8 * (i + 1) + 3); }
    for (i = 0; i < 8; i++) { efree(keep[i]); }
    char *r = (char *) emalloc(40);
    memset(r, 'q', 40);
    r = (char *) erealloc(r, 80);
    memset(r, 'w', 80);
    row_consume(r, 80);
    efree(r);
    char *d = estrndup("abcdefgh", 8);
    row_consume(d, 9);
    efree(d);
    printf("SANITY-OK\n");
})
