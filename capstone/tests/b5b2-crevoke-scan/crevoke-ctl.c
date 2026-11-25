/* crevoke-ctl: positive control for QEMU's CAPSTONE_CREVOKE_SCAN_COUNT. Both arguments malloc and free the same
   buffer. With argv[1] = "1" it also derives four parents inside it, gives them 0, 10, 1500 and 25000 children, and
   ends each parent with one CREVOKE. So the difference between the "1" and the "0" runs is exactly one CREVOKE each
   with a run of 0, 10, 1500 and 25000 records, and nothing else. */
#include <stdio.h>
#include <stdlib.h>
#include <capstone/capability.h>
static const unsigned long kids[4] = {0, 10, 1500, 25000};
int main(int argc, char **argv)
{
    int k = argc > 1 ? atoi(argv[1]) : 0;
    char *p = malloc(4 << 16);
    if (!p) { printf("CREVOKE-CTL malloc failed\n"); return 1; }
    for (int i = 0; k && i < 4; i++) {
        void *q = capstone_cap_derive(p, (unsigned long)i << 16, 1ul << 16);
        for (unsigned long j = 0; j < kids[i]; j++) {
            void *c = capstone_cap_derive(q, j & 0xffff, 1);
            __asm__ volatile("" : : "r"(c));
        }
        capstone_cap_release(q);
        printf("CREVOKE-CTL parent %d children %lu released\n", i, kids[i]);
    }
    free(p);
    printf("CREVOKE-CTL k=%d done\n", k);
    return 0;
}
