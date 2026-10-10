/* The arm's controls, built with the same SDK as the cases and run before any of them.
 *
 * Not cases. Each says what the arm IS: tools/arms.json records what each must do on each
 * configuration (on level0 the read after free COMPLETES, on the Sublet heap it FAULTS; the write
 * one past an object faults on both), and tools/verdicts.py scores no silence from an arm whose
 * controls did otherwise. Both go through plain malloc, as the cases' objects do.
 *
 *   controls.dom bounds-malloc | uaf-malloc
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
  } else {
    fprintf(stderr, "usage: %s bounds-malloc|uaf-malloc\n", argv[0]);
    return 75;
  }
  printf("CONTROL %s RETURNED\n", which);
  return 0;
}
