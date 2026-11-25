/* The kernel facts the trap route's mapping cache rests on (the semantics owner's conditions, 2026-11-25), one mode
 * per run, on a plain mmap'd block (no heap involved). Each mode prints RENEW lines; a fault is the kernel's
 * `sublet: fault cause=` line.
 *
 *   unmap-renewed  mmap, CRENEW, munmap with the renewed reference: must return 0 and the range must leave
 *                  /proc/self/maps. Before the munmap the renewed reference reads the byte written through the old.
 *   unmap-stale    mmap, CRENEW, munmap with the OLD reference: must be refused (nonzero, or a fault) and leave the
 *                  range mapped; then munmap with the renewed one returns 0.
 *   read-stale     mmap, CRENEW, read through the OLD reference: must fault (the old lifetime ended).
 *   many N         mmap, then N CRENEWs, each result used for a store and a load; then munmap with the last: must
 *                  return 0 and unmap the range. With N past the record's generation limit (not read here; the
 *                  run prints how many renewals succeeded) the renewal goes out of place, to another slot, and the
 *                  kernel must still find the block by address. A failed CRENEW (untagged) is reported with its index and ends the loop.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define B (64UL << 10)
static void *renew(void *p)
{
    void *r;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x54, %0, %1, x0" : "=r"(r) : "r"(p) : "memory");
    return r;
}
static unsigned long cursor(void *p) { return __builtin_capstone_cap_get_cursor(p); }
static unsigned long tag(void *p) { return __builtin_capstone_cap_get_tag(p); }
static int mapped(unsigned long a)
{
    FILE *f = fopen("/proc/self/maps", "r");
    char l[512]; unsigned long lo, hi; int in = 0;
    if (!f) return -1;
    while (fgets(l, sizeof l, f)) if (sscanf(l, "%lx-%lx", &lo, &hi) == 2 && a >= lo && a < hi) in = 1;
    fclose(f);
    return in;
}
static char *block(void)
{
    char *p = mmap(0, B, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { printf("RENEW-FAIL mmap errno=%d\n", errno); fflush(stdout); exit(2); }
    p[0] = 0x5a;
    return p;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "unmap-renewed";
    char *p = block();
    unsigned long a = cursor(p);
    printf("RENEW-BEGIN %s base=%#lx mapped=%d\n", mode, a, mapped(a)); fflush(stdout);
    if (!strcmp(mode, "many")) {
        long n = argc > 2 ? atol(argv[2]) : 40000, i;
        char *q = p;
        for (i = 1; i <= n; ++i) {
            q = renew(q);
            if (!tag(q)) break;
            q[i & 4095] = (char)i;
            if (q[i & 4095] != (char)i) { printf("RENEW-FAIL store/load at %ld\n", i); fflush(stdout); return 1; }
        }
        printf("RENEW-MANY renewals=%ld of %ld tag=%lu cursor_same=%d\n", i - 1, n, tag(q), cursor(q) == a);
        fflush(stdout);
        int rc = tag(q) ? munmap(q, B) : munmap(p, B);
        printf("RENEW-UNMAP rc=%d errno=%d mapped=%d\n", rc, rc ? errno : 0, mapped(a)); fflush(stdout);
        return rc || mapped(a) ? 1 : 0;
    }
    char *q = renew(p);
    printf("RENEW-RENEWED tag=%lu cursor_same=%d byte=%#x\n", tag(q), cursor(q) == a, tag(q) ? (unsigned char)q[0] : 0);
    fflush(stdout);
    if (!tag(q)) return 1;
    if (!strcmp(mode, "read-stale")) {
        printf("RENEW-STALE-READ-BEGIN\n"); fflush(stdout);
        volatile char c = p[0];
        printf("RENEW-STALE-READ-RETURNED %#x\n", (unsigned char)c); fflush(stdout);
        return 3;
    }
    if (!strcmp(mode, "unmap-stale")) {
        printf("RENEW-STALE-UNMAP-BEGIN\n"); fflush(stdout);
        int rc = munmap(p, B);
        printf("RENEW-STALE-UNMAP rc=%d errno=%d mapped=%d\n", rc, rc ? errno : 0, mapped(a)); fflush(stdout);
        if (!rc) return 3;
    }
    int rc = munmap(q, B);
    printf("RENEW-UNMAP rc=%d errno=%d mapped=%d\n", rc, rc ? errno : 0, mapped(a)); fflush(stdout);
    return rc || mapped(a) ? 1 : 0;
}
