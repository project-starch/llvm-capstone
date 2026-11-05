/* Arena-cache revocation probe. Phase 1 fills fresh group mappings of one size class, keeps a pointer
 * into the first group, frees everything (mallocng unmaps the groups; the cache keeps them with their
 * group lifetime revoked) and allocates again (the cache hands the mappings back). Phase 2 reads the
 * old pointer: it must fault. Prints one marker per step; the last marker before the fault is the
 * reading. argv[1] = "control" skips the stale read and must print UAF-END. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 4096
static char *a[N], *b[N];
int main(int argc, char **argv)
{
    int control = argc > 1 && !strcmp(argv[1], "control");
    for (int i = 0; i < N; i++) { a[i] = malloc(200); if (!a[i]) { puts("UAF-ENOMEM1"); return 2; } memset(a[i], 0x5a, 200); }
    char *stale = a[N / 2];
    printf("UAF-FILLED %p\n", (void *)stale); fflush(stdout);
    for (int i = 0; i < N; i++) free(a[i]);
    for (int i = 0; i < N; i++) { b[i] = malloc(200); if (!b[i]) { puts("UAF-ENOMEM2"); return 2; } memset(b[i], 0x33, 200); }
    int same = 0; for (int i = 0; i < N; i++) same += (b[i] == stale);
    printf("UAF-REFILLED same-address=%d\n", same); fflush(stdout);
    if (!control) {
        printf("UAF-STALE-READ-BEGIN\n"); fflush(stdout);
        volatile char c = stale[0];
        printf("UAF-STALE-READ-RETURNED %d (temporal safety BROKEN)\n", c); fflush(stdout);
        return 3;
    }
    for (int i = 0; i < N; i++) free(b[i]);
    puts("UAF-END"); return 0;
}
