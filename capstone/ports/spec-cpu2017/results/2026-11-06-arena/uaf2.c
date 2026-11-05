/* Arena-cache safety probe v2 (claim-auditor experiment A). Built against SDKs with heap lifetimes OFF, so a
 * freed object's capability is a SHRINK copy carrying the GROUP's lifetime: only the group's revocation (the cache's
 * CREVOKE, or the kernel's retirement of the mapping) can make a stale pointer fault.
 * Two stale pointers: one into a 200-byte class (its own mapped groups) and one into a 16-byte class (nested groups).
 * Prints the cache counters; a cache arm must show puts > 0 and hits > 0. mode: control | stale200 | stale16 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern unsigned long __capstone_malloc_cache_hits, __capstone_malloc_cache_puts, __capstone_malloc_cache_spills,
                     __capstone_malloc_cache_untracked;
__attribute__((weak)) unsigned long __capstone_malloc_cache_hits, __capstone_malloc_cache_puts,
                     __capstone_malloc_cache_spills, __capstone_malloc_cache_untracked;
#define N 4096
static char *a[N], *b[N], *s[N], *t[N];
int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "control";
    for (int i = 0; i < N; i++) { a[i] = malloc(200); s[i] = malloc(16); if (!a[i] || !s[i]) { puts("UAF-ENOMEM1"); return 2; }
                                  memset(a[i], 0x5a, 200); memset(s[i], 0x5b, 16); }
    char *stale200 = a[N / 2], *stale16 = s[N / 2];
    for (int i = 0; i < N; i++) { free(a[i]); free(s[i]); }
    for (int i = 0; i < N; i++) { b[i] = malloc(200); t[i] = malloc(16); if (!b[i] || !t[i]) { puts("UAF-ENOMEM2"); return 2; }
                                  memset(b[i], 0x33, 200); memset(t[i], 0x34, 16); }
    int same200 = 0, same16 = 0;
    for (int i = 0; i < N; i++) { same200 += (b[i] == stale200); same16 += (t[i] == stale16); }
    printf("UAF-COUNTERS hits=%lu puts=%lu spills=%lu untracked=%lu same200=%d same16=%d\n", __capstone_malloc_cache_hits,
           __capstone_malloc_cache_puts, __capstone_malloc_cache_spills, __capstone_malloc_cache_untracked, same200, same16);
    fflush(stdout);
    if (!strcmp(mode, "stale200") || !strcmp(mode, "stale16")) {
        char *p = !strcmp(mode, "stale200") ? stale200 : stale16;
        printf("UAF-STALE-READ-BEGIN %s\n", mode); fflush(stdout);
        volatile char c = p[0];
        printf("UAF-STALE-READ-RETURNED %s 0x%02x\n", mode, (unsigned char)c); fflush(stdout);
        return 3;
    }
    puts("UAF-END"); return 0;
}
