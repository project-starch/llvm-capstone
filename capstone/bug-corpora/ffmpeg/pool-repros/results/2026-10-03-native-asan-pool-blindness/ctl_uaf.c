/* Positive control for the INSTRUMENT: a plain heap use-after-free that ASan must
 * report. If this comes back clean, ASan is not active and no clean result in this
 * experiment means anything. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void)
{
    unsigned char *p = malloc(32);
    memset(p, 0xA0, 32);
    free(p);
    printf("stale_read=0x%02X\n", p[0]);   /* must be heap-use-after-free */
    puts("DONE");
    return 0;
}
