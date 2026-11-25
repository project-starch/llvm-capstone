/* untag-ctl: positive control for QEMU's CAPSTONE_UNTAGGED_ARITH_COUNT. With argv[1] = "1" it executes exactly
   1000 CINCOFFSETIMM, 500 CINCOFFSET and 250 SCC on an UNTAGGED x[rs1] (integer registers), and 300
   CINCOFFSETIMM on a TAGGED x[rs1] (a real pointer) that must not count. With "0" it runs none of them, so the
   difference between the two runs is exactly what the loops did. */
#include <stdio.h>
#include <stdlib.h>
static char buf[64];
int main(int argc, char **argv)
{
    int k = argc > 1 ? atoi(argv[1]) : 0;
    unsigned long acc = 0;
    for (int i = 0; i < 1000 * k; i++) {
        unsigned long r, z = (unsigned long)i;
        __asm__ volatile(".insn i 0x5b, 2, %0, %1, 8" : "=r"(r) : "r"(z));
        acc += r;
    }
    for (int i = 0; i < 500 * k; i++) {
        unsigned long r, z = (unsigned long)i, off = 16;
        __asm__ volatile(".insn r 0x5b, 1, 0x0c, %0, %1, %2" : "=r"(r) : "r"(z), "r"(off));
        acc += r;
    }
    for (int i = 0; i < 250 * k; i++) {
        unsigned long r, z = (unsigned long)i, cur = 0x1000;
        __asm__ volatile(".insn r 0x5b, 1, 0x05, %0, %1, %2" : "=r"(r) : "r"(z), "r"(cur));
        acc += r;
    }
    char *p = buf;
    for (int i = 0; i < 300 * k; i++) {
        char *q;
        __asm__ volatile(".insn i 0x5b, 2, %0, %1, 1" : "=r"(q) : "r"(p));
        acc += (unsigned long)(q - p);
    }
    printf("UNTAG-CTL k=%d acc=%lu\n", k, acc);
    return 0;
}
