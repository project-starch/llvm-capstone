/* Live-peak control (2026-10-22): the record counter must read what these programs hold.
 *   hold N:  N malloc(32) objects live at once, then all freed. Peak live >= N; minted >= N.
 *   churn N: N rounds of malloc(32)/free. Minted >= N; peak live stays near the baseline. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  if (argc != 3) return 64;
  long n = atol(argv[2]);
  if (!strcmp(argv[1], "hold")) {
    void **v = malloc(n * sizeof *v);
    if (!v) return 70;
    for (long i = 0; i < n; ++i) if (!(v[i] = malloc(32))) { printf("HOLD-NULL at %ld\n", i); return 71; }
    printf("HOLD-PEAK %ld\n", n);
    for (long i = 0; i < n; ++i) free(v[i]);
    free(v);
    printf("HOLD-OK %ld\n", n);
    return 0;
  }
  if (!strcmp(argv[1], "churn")) {
    for (long i = 0; i < n; ++i) { volatile long *p = malloc(32); if (!p) { printf("CHURN-NULL at %ld\n", i); return 71; } p[0] = i; free((void *)p); }
    printf("CHURN-OK %ld\n", n);
    return 0;
  }
  return 64;
}
