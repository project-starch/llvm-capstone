/* POSITIVE CONTROL FOR THE VEHICLE, not a corpus case.
 *
 * Every verdict this harness has produced so far is a MISS. Without a case CHERI
 * demonstrably CATCHES, "CHERI misses A1" cannot be told from "this harness never
 * reports anything". So: a plain heap-buffer-overflow on a malloc'd buffer, which
 * is the one thing purecap must stop -- the bounds come straight from malloc.
 *
 * Prints a marker before and after. Reaching the AFTER line means CHERI did NOT
 * stop it, which would invalidate every MISS in the table.
 */
#include <stdio.h>
#include <stdlib.h>

int
main(void)
{
    volatile char *p = malloc(64);
    int i;

    if (!p) { printf("CONTROL=ALLOC-FAILED\n"); return 3; }
    printf("CONTROL=BEFORE\n");
    fflush(stdout);
    /* Walk well past the end. On purecap the bounds are exactly 64. */
    for (i = 0; i < 4096; i++)
        p[i] = (char)i;
    printf("CONTROL=AFTER value=%d\n", (int)p[4095]);
    fflush(stdout);
    return 0;
}
