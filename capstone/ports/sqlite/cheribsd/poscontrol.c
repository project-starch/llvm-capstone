/* poscontrol.c -- the positive control the Sublet gate requires.
 *
 * Sublet 5.1: "each CheriBSD run carries its own positive control in the same
 * boot, which faults while the cases complete." Table 10 (<Dc>): the count is
 * only admissible "after the mechanism's own positive control passes".
 *
 * This is a use-after-free through the SYSTEM allocator, i.e. the one layer
 * CheriBSD's heap revocation does observe. With revocation enabled it MUST
 * fault (SIGPROT). If it instead runs to completion, revocation is not active
 * and every corpus result from that boot is void -- a zero would be
 * indistinguishable from a misconfigured environment.
 *
 * Exits 0 only if the dangling access succeeded, which is the FAILURE case
 * for this program.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  puts("poscontrol BEGIN");

  /* Large enough to come from the system allocator rather than any cache. */
  const size_t n = 4096;
  volatile unsigned long *p = malloc(n);
  if (!p) { puts("poscontrol ERROR malloc"); return 2; }
  for (size_t i = 0; i < n / sizeof(*p); i++) p[i] = 0x5A5A5A5A5A5A5A5AUL;
  unsigned long before = p[0];
  printf("poscontrol before=%lu\n", before);

  free((void *)p);

  /* Churn so a quarantine threshold is crossed and a revocation sweep runs.
     Without this, a default (non-zero) quarantine may not have swept yet and
     the dangling load would still be allowed -- that is use-after-REALLOCATION
     semantics, not use-after-free, and is exactly the distinction 3.1 of the
     PoisonCap paper draws. */
  for (int i = 0; i < 4096; i++) { void *q = malloc(n); if (q) free(q); }

  /* The dangling load. Under revocation this traps and we never return. */
  unsigned long after = p[0];
  printf("poscontrol after=%lu\n", after);
  puts("poscontrol NOTRAP done  <-- REVOCATION NOT ACTIVE, results are void");
  return 0;
}
