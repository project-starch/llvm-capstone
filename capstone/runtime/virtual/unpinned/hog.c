/* Fragment guest memory so compaction has to move the domain's pages.
 * usage: hog MB  -- maps and touches MB MiB, unmaps every other 4 MiB chunk (holes the domain's
 * pages then land in), prints READY and sleeps until killed; killing it frees the rest. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
int main(int argc, char **argv) {
    size_t mb = argc > 1 ? strtoul(argv[1], 0, 0) : 1024, chunk = 4u << 20, bytes = mb << 20;
    char *p = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 1; }
    for (size_t i = 0; i < bytes; i += 4096) p[i] = 1;
    for (size_t i = 0; i + chunk <= bytes; i += 2 * chunk) munmap(p + i, chunk);
    printf("READY\n"); fflush(stdout);
    for (;;) pause();
}
