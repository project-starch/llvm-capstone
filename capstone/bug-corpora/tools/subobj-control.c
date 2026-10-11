/* The positive control for a sub-object-bounds arm: a write one past an 8-byte array FIELD, into
 * its sibling field, inside one malloc'd struct. Built with the arm's flags it must die by SIGPROT
 * (status 162 under the CheriBSD runner); built without them it completes, as every allocation-
 * granular arm must. Through a variable index, so no rule about constant indices applies. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct pair { unsigned char a[8]; unsigned char b[8]; };
int main(int argc, char **argv) {
  struct pair *s = malloc(sizeof *s);
  if (!s) return 75;
  memset(s, 0, sizeof *s);
  volatile int i = 8 + (argc > 99);   /* 8, not foldable */
  /* A line BEFORE the access, flushed: the runner matches stdout, and a control that dies before
   * printing anything would read as a FAILED row even when it faulted exactly as required. */
  printf("SUBOBJ-CONTROL BEGIN\n");
  fflush(stdout);
  s->a[i] = 0x5a;                     /* one past a[], onto b[0] */
  printf("SUBOBJ-CONTROL RETURNED b0=0x%02x\n", s->b[0]);
  (void)argv;
  return 0;
}
