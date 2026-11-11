/* Keep enough groups live to grow mallocng metadata, then verify and release. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static unsigned char *objects[16384];
int main(int argc, char **argv)
{
    unsigned n = argc > 1 ? strtoul(argv[1], 0, 10) : 1024;
    if (!n || n > 16384) return 2;
    unsigned long sum = 0;
#ifdef METADATA_PROFILE
    __asm__ volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory");
#endif
    for (unsigned round = 0; round < 2; round++) {
        for (unsigned i = 0; i < n; i++) {
            size_t size = 64 + (i % 32) * 128;
            objects[i] = malloc(size);
            if (!objects[i]) return 3;
            objects[i][0] = (unsigned char)i;
            objects[i][size-1] = (unsigned char)(i >> 8);
        }
        for (unsigned i = 0; i < n; i++) {
            size_t size = 64 + (i % 32) * 128;
            if (objects[i][0] != (unsigned char)i ||
                objects[i][size-1] != (unsigned char)(i >> 8)) return 4;
            sum += objects[i][0] + objects[i][size-1];
            free(objects[i]);
        }
    }
#ifdef METADATA_PROFILE
    __asm__ volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory");
#endif
    printf("PASS metadata-break n=%u sum=%lu\n", n, sum);
    return 0;
}
