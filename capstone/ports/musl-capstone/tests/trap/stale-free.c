/* A reference kept after free: free ends the object's lifetime (CREVOKE), so the read through
 * the old reference must fault with cause 25 (stale) and the kernel kills the program. Reaching
 * the end means the object outlived free: the program prints that and exits 1. */
#include <stdlib.h>
#include <unistd.h>

int main(void)
{
    volatile char *p = malloc(32);
    if (!p) return 2;
    p[0] = 42;
    if (p[0] != 42) return 3;
    free((void *)p);
    write(1, "stale-free: reading after free\n", 31);
    char c = p[0];
    write(1, "stale-free: FAIL, no fault\n", 27);
    return c == 42 ? 1 : 4;
}
