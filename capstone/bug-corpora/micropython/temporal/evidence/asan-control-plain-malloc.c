/* Positive control for the ASan measurement: an ordinary malloc use-after-free.
   If ASan does not report THIS, then its silence on MicroPython's bytearray
   says nothing about MicroPython. */
#include <stdlib.h>
#include <stdio.h>
int main(void) {
    char *p = malloc(64);
    p[0] = 'x';
    free(p);
    p[0] = 'y';            /* use after free */
    printf("control still running: %c\n", p[0]);
    return 0;
}
