/* tsapp-stress-red.c (nested lane, 2026-11-25): M-16's stress form (tsapp-stress.c) with its first two reductions in
 * one image, chosen by a mode argument, because the firmware cap fits one such image per boot. NOT a port file. Built
 * and linked as tsapp-stress.c is (tshark.c.o's compile line, tshark_m5's link, tsapp_stress.o in tshark.c.o's place).
 *   a  tsapp-stress.c's loop unchanged: a BLOCK_FAST packet scope and a BLOCK file scope (one allocation in 12, half
 *      of those freed again, the file scope reset every 1,000 packets), LCG sizes. Its STRESS lines equal
 *      tsapp-stress.c's for the same arguments (the positive control: P1 stopped in this loop once).
 *   f  reduction 1: BLOCK_FAST alone. Every allocation goes to the packet scope, with the same LCG sizes; no file scope.
 *   s  reduction 2: BLOCK_FAST alone at one fixed size, 32 bytes.
 * Each allocation's first and last bytes are written and read back into a checksum, so the output is a function of
 * argv only and the twin's lines are the reference.
 * argv: mode packets K report seed. Prints "STRESS begin", a heartbeat "STRESS pkt=<i> fast=<n> block=<m> free=<f>
 * sum=<x>" every <report> packets, and "STRESS done ...". rc 0; 2 on bad arguments. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (argc != 6 || strlen(argv[1]) != 1 || !strchr("afs", argv[1][0])) {
        fprintf(stderr, "usage: tsapp-stress-red a|f|s packets K report seed\n"); return 2;
    }
    const char mode = argv[1][0];
    unsigned long packets = strtoul(argv[2], NULL, 10), k = strtoul(argv[3], NULL, 10);
    unsigned long report = strtoul(argv[4], NULL, 10);
    lcg = strtoull(argv[5], NULL, 10);
    if (!packets || !k || !report) { fprintf(stderr, "tsapp-stress-red: zero argument\n"); return 2; }
    setvbuf(stdout, NULL, _IOLBF, 0);
    wmem_allocator_t *pkt = wmem_allocator_new(WMEM_ALLOCATOR_BLOCK_FAST);
    wmem_allocator_t *file = mode == 'a' ? wmem_allocator_new(WMEM_ALLOCATOR_BLOCK) : NULL;
    unsigned long long sum = 0, nfast = 0, nblock = 0, nfree = 0;
    void *keep = NULL;
    printf("STRESS begin mode=%c packets=%lu K=%lu report=%lu seed=%s\n", mode, packets, k, report, argv[5]);
    for (unsigned long i = 1; i <= packets; i++) {
        for (unsigned long j = 0; j < k; j++) {
            size_t n = mode == 's' ? 32 : pick();
            int to_file = mode == 'a' && rnd() % 12 == 0;
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
        if (file && i % 1000 == 0) { wmem_free_all(file); keep = NULL; }
        if (i % report == 0)
            printf("STRESS pkt=%lu fast=%llu block=%llu free=%llu sum=%016llx\n", i, nfast, nblock, nfree, sum);
    }
    wmem_destroy_allocator(pkt);
    if (file) wmem_destroy_allocator(file);
    printf("STRESS done packets=%lu fast=%llu block=%llu free=%llu sum=%016llx\n", packets, nfast, nblock, nfree, sum);
    return 0;
}
