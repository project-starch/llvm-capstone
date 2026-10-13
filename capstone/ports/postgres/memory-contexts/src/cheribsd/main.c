/* CheriBSD allocator replay. Layout differs from x86: recorded backing counts
 * are reference observations, not required outcomes of this ABI. */
#include "postgres.h"
#include "replay-engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Noreturn void replay_die(const char *what) {
  fprintf(stderr, "PG_REPLAY failed: %s\n", what);
  exit(1);
}
_Noreturn void replay_die_at(unsigned long i, const char *what, unsigned long id) {
  fprintf(stderr, "PG_REPLAY record=%lu id=%lu: %s\n", i, id, what);
  exit(1);
}
void *replay_alloc(size_t n) { return calloc(1, n); }
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  FILE *file = fopen(argv[1], "rb");
  if (!file || fseek(file, 0, SEEK_END))
    return 2;
  long length = ftell(file);
  if (length < (long)sizeof(struct a11_head) || length > (64L << 20) ||
      fseek(file, 0, SEEK_SET))
    return 2;
  unsigned char *bytes = malloc((size_t)length);
  if (!bytes || fread(bytes, 1, length, file) != (size_t)length ||
      ferror(file) || fgetc(file) != EOF || fclose(file))
    return 2;
  struct a11_head *h = (void *)bytes;
  if (memcmp(h->magic, A11_MAGIC, 8) || h->version != A11_VERSION ||
      h->recsize != sizeof(struct a11_rec) || h->endian != A11_ENDIAN ||
      h->ppid || ((size_t)length - sizeof *h) % sizeof(struct a11_rec))
    return 2;
  struct replay_counts counts = {0};
  replay_run((void *)(bytes + sizeof *h),
             ((size_t)length - sizeof *h) / sizeof(struct a11_rec), &counts);
  printf("PG_REPLAY PASS creates=%lu allocs=%lu frees=%lu reallocs=%lu "
         "resets=%lu deletes=%lu checked=%lu pointer_bytes=%zu\n",
         counts.create, counts.alloc, counts.free, counts.realloc,
         counts.reset, counts.delete, counts.checked, sizeof(void *));
  free(bytes);
  return 0;
}
