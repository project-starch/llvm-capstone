/* Safety probe for CAPSTONE_HEAP_MAPPING_REUSE (malloc-mapping.c). Build against an SDK with object lifetimes OFF
 * (-DCAPSTONE_HEAP_LIFETIMES=OFF), so a freed object's capability carries only its mapping owner's lifetime: only the
 * reuse path's CREVOKE, or the kernel's retirement of the mapping, can make a stale pointer fault.
 * Two stale pointers: one into a 200-byte class and one into a 16-byte class. The counters show whether mappings
 * were cached and reissued (puts, hits) and whether the stale slot's address came back (same200, same16).
 * mode: control | stale200 | stale16. Expected: stale reads fault (cause 25) with or without the switch, and RETURN
 * the new owner's byte with CAPSTONE_HEAP_MAPPING_REUSE_TEST_NO_REVOKE (the positive control). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern unsigned long __heap_reuse_hits, __heap_reuse_puts, __heap_reuse_spills,
                     __heap_reuse_untracked;
__attribute__((weak)) unsigned long __heap_reuse_hits, __heap_reuse_puts,
                     __heap_reuse_spills, __heap_reuse_untracked;
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
    printf("UAF-COUNTERS hits=%lu puts=%lu spills=%lu untracked=%lu same200=%d same16=%d\n", __heap_reuse_hits,
           __heap_reuse_puts, __heap_reuse_spills, __heap_reuse_untracked, same200, same16);
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
