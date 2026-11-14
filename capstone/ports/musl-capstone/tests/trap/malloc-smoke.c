#include <stdlib.h>
int main(void)
{
    char *p = malloc(32);
    if (!p) return 1;
    *(volatile char *)p = 42;
    if (*(volatile char *)p != 42) return 2;
    free(p);
    return 7;
}
