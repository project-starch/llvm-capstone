/* tsapp-stress.c (nested lane, 2026-11-25): an M-16 stress form, NOT a port file. Built with tshark.c.o's own compile
 * line and linked exactly as tshark_m5 is (the same wsutil, glib and runtime objects, so the same wmem 0001 block
 * allocators), in tshark.c.o's place.
 *
 * M-16: on P1 the protected tshark does not return in about 4 boots of 10, each time held at the CDERIVE of
 * wmem_block_fast_sublet_alloc. tshark issues about 1.9 M of those per run (twin zones, test capture). This program
 * issues the same allocation path many times in one process, the way tshark's scopes do:
 *   - a packet scope (BLOCK_FAST): per packet, K allocations, then wmem_free_all;
 *   - a file scope (BLOCK): one allocation in 12 goes there, and half of those are freed again with wmem_free;
 *     the file scope is reset every 1,000 packets.
 * Sizes come from a fixed LCG: 70 % 8-64 bytes, 25 % 65-512, 5 % 513-4096. Each allocation's first and last bytes are
 * written and read back into a checksum, so the run's checksum is a function of argv only and the twin's value is the
 * reference for the board's.
 *
 * argv: packets K report seed. Prints "STRESS begin", "STRESS pkt=<i> fast=<n> block=<m> sum=<x>" every <report>
 * packets (a heartbeat: a stop is placed to within <report> packets), and "STRESS done ..." at the end. rc 0. */
#include <stdio.h>
#include <stdlib.h>

#include <glib.h>
#include <wsutil/wmem/wmem.h>

static unsigned long long lcg;
static unsigned rnd(void) { lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL; return (unsigned)(lcg >> 33); }

static size_t pick(void)
{
    unsigned r = rnd() % 100, x = rnd();
    if (r < 70) return 8 + x % 57;
    if (r < 95) return 65 + x % 448;
    return 513 + x % 3584;
}

int main(int argc, char **argv)
{
    if (argc != 5) { fprintf(stderr, "usage: tsapp-stress packets K report seed\n"); return 2; }
    unsigned long packets = strtoul(argv[1], NULL, 10), k = strtoul(argv[2], NULL, 10);
    unsigned long report = strtoul(argv[3], NULL, 10);
    lcg = strtoull(argv[4], NULL, 10);
    if (!packets || !k || !report) { fprintf(stderr, "tsapp-stress: zero argument\n"); return 2; }
    setvbuf(stdout, NULL, _IOLBF, 0);
    wmem_allocator_t *pkt = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK_FAST);
    wmem_allocator_t *file = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK);
    unsigned long long sum = 0, nfast = 0, nblock = 0, nfree = 0;
    void *keep = NULL;
    printf("STRESS begin packets=%lu K=%lu report=%lu seed=%s\n", packets, k, report, argv[4]);
    for (unsigned long i = 1; i <= packets; i++) {
        for (unsigned long j = 0; j < k; j++) {
            size_t n = pick();
            int to_file = rnd() % 12 == 0;
            volatile unsigned char *p = wmem_alloc(to_file ? file : pkt, n);
            p[0] = (unsigned char)(i + j);
            p[n - 1] = (unsigned char)(n);
            sum = sum * 1099511628211ULL + p[0] + ((unsigned long long)p[n - 1] << 8) + n;
            if (to_file) {
                nblock++;
                if (rnd() & 1) {
                    if (keep) { wmem_free(file, keep); nfree++; }
                    keep = (void *)p;
                }
            } else {
                nfast++;
            }
        }
        wmem_free_all(pkt);
        if (i % 1000 == 0) { wmem_free_all(file); keep = NULL; }
        if (i % report == 0)
            printf("STRESS pkt=%lu fast=%llu block=%llu free=%llu sum=%016llx\n", i, nfast, nblock, nfree, sum);
    }
    wmem_destroy_allocator(pkt);
    wmem_destroy_allocator(file);
    printf("STRESS done packets=%lu fast=%llu block=%llu free=%llu sum=%016llx\n", packets, nfast, nblock, nfree, sum);
    return 0;
}
