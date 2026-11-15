#include <stdio.h>
#include <stdlib.h>
/* The protected fault control: a read after free. mallocng ends the object lifetime at free, so the read must fault (cause 25). */
int main(void)
{
    volatile int *p = malloc(64);
    p[0] = 7;
    free((void *)p);
    printf("GATE-FAULT before the stale read\n");
    fflush(stdout);
    return p[0] == 7 ? 0 : 1;
}
