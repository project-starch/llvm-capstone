/* Each request must return once, with its own result, across supervisor stops. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include "vm.h"

int main(void)
{
    unsigned char *p = malloc(64);
    if (!p) return 1;
    for (unsigned i = 0; i < 64; ++i) {
        p[i] = i;
        if (__capstone_vm_nodes(i & 1 ? 8192 : 0) != 0 ||
            __capstone_vm_nodes(~0UL) != -ENOMEM || p[i] != i) {
            puts("NODE_SERVICES_FAIL");
            return 1;
        }
    }
    free(p);
    puts("NODE_SERVICES_OK requests=128");
    return 0;
}
