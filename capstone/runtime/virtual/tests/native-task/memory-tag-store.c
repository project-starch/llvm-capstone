/* SPDX-License-Identifier: MIT */
/* A capability store must itself COW a page first mapped by a read. */
#include <stdio.h>
#include <sys/mman.h>
int main(void)
{
    unsigned char target[256];
    for (unsigned i=0; i<256; ++i) target[i] = (i%251)+1;
    volatile unsigned char *p = mmap(0, 256*4096, PROT_READ|PROT_WRITE,
                                    MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 1;
    for (unsigned i=0; i<256; ++i) if (p[i*4096]) return 2;
    for (unsigned i=0; i<256; ++i) {
        unsigned char * volatile *slot = (void *)(p+i*4096);
        /* Earlier stores must not modify any untouched shared-zero page. */
        for (unsigned j=0; j<16; ++j) if (p[i*4096+j]) return 3;
        *slot = &target[i];
    }
    /* Distinct pointers plus a delayed pass expose accidental page aliasing. */
    for (unsigned i=0; i<256; ++i) {
        unsigned char * volatile *slot = (void *)(p+i*4096);
        if (!__builtin_capstone_cap_get_tag(*slot) || *slot != &target[i] ||
            **slot != (i%251)+1) return 4;
    }
    if (munmap((void *)p, 256*4096)) return 5;
    puts("TAG_FIRST_PASS");
    return 0;
}
