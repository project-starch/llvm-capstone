/* lq-hog MB: fragment guest memory for lqmicro's compact mode (plain program, run in the
 * background before it, outside any window). Maps and touches MB MiB, unmaps every other 4 MiB
 * chunk, so the pages the benchmark maps next land in the holes, prints READY, and exits when
 * /tmp/lq-hog-exit appears (the benchmark creates it after its touch), freeing the rest: the
 * benchmark's pages then sit between free blocks and compaction has to move them. */
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? strtoul(argv[1], 0, 0) : 1024, chunk = 4UL << 20, bytes = mb << 20;
    char *p = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    for (size_t i = 0; i < bytes; i += 4096) p[i] = 1;
    for (size_t i = 0; i + chunk <= bytes; i += 2 * chunk) munmap(p + i, chunk);
    unlink("/tmp/lq-hog-exit");
    printf("READY\n");
    fflush(stdout);
    struct timespec t = {0, 50 * 1000 * 1000};
    while (access("/tmp/lq-hog-exit", F_OK)) nanosleep(&t, 0);
    return 0;
}
