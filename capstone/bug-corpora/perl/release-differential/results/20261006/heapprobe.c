/* Two questions this arm's verdict depends on, neither answered by the eval/shim
   controls:
     1. does a free in THIS image actually revoke, so a system-allocator
        use-after-free can fault at all?  (If not, "sublet is silent" says nothing.)
     2. does realloc MOVE a grown block, so 254b30e378's dangling vbuf can exist?
        Our level0 grows within slack and keeps the old bounds; host glibc moved it,
        which is why ASan saw the use-after-free there. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  /* 2 first, because it must not fault and 1 probably will. */
  char *p = malloc(26);
  memset(p, 'a', 26);
  char *q = realloc(p, 52);
  printf("REALLOC old=%p new=%p moved=%d\n", (void *)p, (void *)q, p != q);
  fflush(stdout);
  /* The same shape at Perl's own sizes: SvGROW from 26 to 52 is what the case does. */
  char *r = malloc(1000);
  char *s = realloc(r, 4000);
  printf("REALLOC2 moved=%d\n", r != s);
  fflush(stdout);
  /* 1: the positive control. argv keeps the index opaque so the read cannot be
     folded away, and the value is printed so the load really happens. */
  size_t n = (size_t)(argc > 1 ? atoi(argv[1]) : 0);
  char *v = malloc(64);
  memset(v, 'Z', 64);
  free(v);
  printf("ABOUT-TO-READ-FREED\n");
  fflush(stdout);
  volatile char c = v[n];
  printf("READ-FREED-OK got=%c -- free did NOT revoke\n", c);
  return 0;
}
