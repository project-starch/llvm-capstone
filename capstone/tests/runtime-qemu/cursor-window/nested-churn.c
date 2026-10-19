/* Provokes S-SHRINK (the runtime lane's bind_slots finding): creates and frees mallocng nested groups for as long as
 * it runs, so a quantum preemption can land while bind_slots holds the group capability with its cursor outside its
 * window. argv[1] = rounds. Prints NC_DONE rounds=N on completion; a refused SHRINK kills the process (cause 29). */
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv)
{
    unsigned long rounds = argc > 1 ? strtoul(argv[1], 0, 0) : 20000, s = 12345, touched = 0;
    void *p[96];
    for (unsigned long r = 0; r < rounds; r++) {
        for (int i = 0; i < 96; i++) {
            s = s * 6364136223846793005UL + 1442695040888963407UL;
            size_t sz = 16 + (size_t)((s >> 33) % 1200);      /* many small size classes */
            p[i] = malloc(sz);
            if (p[i]) { ((volatile char *)p[i])[0] = (char)i; touched++; }
        }
        for (int i = 0; i < 96; i++) free(p[i]);
        if ((r + 1) % 5000 == 0) { printf("NC_PROGRESS rounds=%lu\n", r + 1); fflush(stdout); }
    }
    printf("NC_DONE rounds=%lu touched=%lu\n", rounds, touched);
    return 0;
}
