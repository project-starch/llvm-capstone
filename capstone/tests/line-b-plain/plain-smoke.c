/* plain-smoke: the Plain images' loader and libc control (the trap route's smoke.dom prints "ok" and exits 7). It
   also takes and frees one small and one mmapped allocation through Plain-matched musl's mallocng. Exit 7 on
   success, 1 otherwise. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void)
{
    char *a = malloc(100), *b = malloc(1 << 20);
    if (!a || !b) return 1;
    memset(a, 1, 100); memset(b, 2, 1 << 20);
    int ok = a[99] == 1 && b[(1 << 20) - 1] == 2;
    free(a); free(b);
    printf("ok\n");
    return ok ? 7 : 1;
}
