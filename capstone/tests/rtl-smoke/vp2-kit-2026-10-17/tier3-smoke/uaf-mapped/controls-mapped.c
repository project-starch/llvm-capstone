/* The arm's controls, built with the same SDK as the cases and run before any of them.
 *
 * Not cases. Each says what the arm IS: tools/arms.json records what each must do on each
 * configuration (on level0 the read after free COMPLETES, on virtual mallocng it FAULTS; the write
 * one past an object faults on both), and tools/verdicts.py scores no silence from an arm whose
 * controls did otherwise. Both go through plain malloc, as the cases' objects do.
 *
 *   controls.dom bounds-malloc | uaf-malloc | uaf-mapped
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__attribute__((noinline, used)) void control_write_probe(volatile unsigned char *p) { *p = 1; }
__attribute__((noinline, used)) unsigned control_read_probe(const volatile unsigned char *p) { return *p; }

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  const char *which = argc == 2 ? argv[1] : "";
  if (!strcmp(which, "bounds-malloc")) {
    unsigned char *p = malloc(16);
    if (!p) return 75;
    printf("CONTROL %s mark\n", which);
    control_write_probe(p + 16);
  } else if (!strcmp(which, "uaf-malloc")) {
    unsigned char *p = malloc(32);
    if (!p) return 75;
    p[0] = 7;
    free(p);
    printf("CONTROL %s mark\n", which);
    (void)control_read_probe(p);
  } else if (!strcmp(which, "uaf-mapped")) {
    /* uaf-malloc with the freed object's page kept MAPPED (vp2 board lane, 2026-10-22): two live 32-byte neighbours
     * from the same size class hold mallocng's group, so free(p) cannot unmap it. A read that returns prints
     * STALE_READ_SURVIVED: a load through a freed (revoked) capability that completed. */
    unsigned char *keep = malloc(32), *p = malloc(32), *keep2 = malloc(32);
    if (!keep || !p || !keep2) return 75;
    keep[0] = 1; p[0] = 7; keep2[0] = 2;
    unsigned long kp = (unsigned long)keep >> 12, pp = (unsigned long)p >> 12, k2p = (unsigned long)keep2 >> 12;
    free(p);
    printf("CONTROL %s mark same_page_keep=%d same_page_keep2=%d\n", which, kp == pp, k2p == pp);
    unsigned v = control_read_probe(p);
    printf("STALE_READ_SURVIVED value=%u\n", v);
    printf("CONTROL %s live %u %u\n", which, keep[0], keep2[0]);
  } else {
    fprintf(stderr, "usage: %s bounds-malloc|uaf-malloc|uaf-mapped\n", argv[0]);
    return 75;
  }
  printf("CONTROL %s RETURNED\n", which);
  return 0;
}
