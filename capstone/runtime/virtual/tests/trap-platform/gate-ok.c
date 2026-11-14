#include <stdio.h>
#include <stdlib.h>
/* The protected control that must exit 0: allocate, use, free. */
int main(void)
{
    int *p = malloc(64);
    p[0] = 7;
    printf("GATE-OK %d\n", p[0]);
    free(p);
    return 0;
}
