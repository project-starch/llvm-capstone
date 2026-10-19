/* read-stale: does a domain see the bytes a delegated read() just wrote? Each round maps a fresh region, touches it
 * as the arm says, read()s a known pattern file into it through the delegate, then checks every byte from the domain.
 *   capstone-vexec read-stale.dom FILE ARM ROUNDS PAGES
 * ARM: 1 = read-touch each page first (Linux maps the zero page; the read's copy then COWs it)
 *      2 = write-touch each page first (a private page exists before the read)
 *      3 = untouched (the kernel faults the pages in during the copy)
 * Prints RS_ROUND bad=<bytes that differ> first_bad=<offset> first_val=<value> per bad round, and RS_DONE totals.
 * A wrong answer here is the jump-to-0 made visible: a domain reading a stale zero page where data was written. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static unsigned char expect(unsigned long i) { return (unsigned char)(0x5a ^ (i * 131u) ^ (i >> 12)); }

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: read-stale FILE ARM ROUNDS PAGES\n"); return 2; }
    const char *path = argv[1];
    int arm = atoi(argv[2]);
    unsigned long rounds = strtoul(argv[3], 0, 0), pages = strtoul(argv[4], 0, 0), len = pages * 4096;
    unsigned long bad_rounds = 0, bad_bytes = 0, zero_bytes = 0;
    volatile unsigned char sink = 0;
    printf("RS_BEGIN arm=%d rounds=%lu pages=%lu\n", arm, rounds, pages);
    for (unsigned long r = 0; r < rounds; ++r) {
        unsigned char *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { printf("RS_FAIL mmap round=%lu\n", r); return 3; }
        for (unsigned long g = 0; g < pages; ++g) {
            if (arm == 1) sink ^= p[g * 4096];
            else if (arm == 2) p[g * 4096] = 0xee;
        }
        int fd = open(path, O_RDONLY);
        if (fd < 0) { printf("RS_FAIL open round=%lu\n", r); return 3; }
        unsigned long got = 0;
        while (got < len) {
            long n = read(fd, p + got, len - got);
            if (n <= 0) break;
            got += (unsigned long)n;
        }
        close(fd);
        if (got != len) { printf("RS_FAIL short read round=%lu got=%lu\n", r, got); return 3; }
        unsigned long bad = 0, first = 0; unsigned char firstv = 0;
        for (unsigned long i = 0; i < len; ++i)
            if (p[i] != expect(i)) { if (!bad) { first = i; firstv = p[i]; } ++bad; zero_bytes += p[i] == 0; }
        if (bad) {
            ++bad_rounds; bad_bytes += bad;
            printf("RS_ROUND round=%lu bad=%lu first_bad=%lu first_val=0x%02x\n", r, bad, first, firstv);
        }
        munmap(p, len);
    }
    printf("RS_DONE arm=%d rounds=%lu bad_rounds=%lu bad_bytes=%lu zero_bytes=%lu sink=%u\n",
           arm, rounds, bad_rounds, bad_bytes, zero_bytes, (unsigned)sink);
    return 0;
}
