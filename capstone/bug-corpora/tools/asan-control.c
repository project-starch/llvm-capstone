/* The positive control for tools/run-native-asan.py: an access ASan MUST report, on a heap block
 * of the size the corpus's own allocator takes from malloc, built with the same compiler and flags.
 *
 *   asan-control past <bytes>   malloc(bytes), then read one byte past it  -> heap-buffer-overflow
 *   asan-control uaf  <bytes>   malloc(bytes), free it, read it            -> heap-use-after-free
 *
 * A corpus whose cases are silent under ASan says something only if this reports in the same run:
 * then the silence is the arena (no redzone inside one block, no free() on a recycled object), not
 * an ASan that was never armed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__attribute__((noinline)) static unsigned char touch(const volatile unsigned char *p) { return *p; }

int main(int argc, char **argv) {
  if (argc != 3)
    return 75;
  size_t n = (size_t)strtoull(argv[2], NULL, 0);
  unsigned char *p = malloc(n);
  if (!p)
    return 75;
  memset(p, 0x5a, n);
  if (!strcmp(argv[1], "past")) {
    printf("asan-control past %zu: %u\n", n, touch(p + n));
  } else if (!strcmp(argv[1], "uaf")) {
    free(p);
    printf("asan-control uaf %zu: %u\n", n, touch(p));
  } else {
    return 75;
  }
  return 0; /* reached only if ASan did not stop the program */
}
