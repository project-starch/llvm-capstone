/* CheriBSD allocator replay. Layout differs from x86: recorded backing counts
 * are reference observations, not required outcomes of this ABI. */
#include "postgres.h"
#include "replay-engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef PG_POISONCAP
#include "poisoncap.h"
#endif

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
  if (argc != 2
#ifdef PG_POISONCAP
      && argc != 3
#endif
  )
    return 2;
#ifdef PG_POISONCAP
  unsigned mode = 1;
  if (argc == 3) {
    if (strcmp(argv[2], "0") && strcmp(argv[2], "1")) return 2;
    mode = argv[2][0] - '0';
  }
  pg_poisoncap_init(mode);
#endif
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
#ifdef PG_POISONCAP
  pg_poisoncap_report();
#endif
  return 0;
}
