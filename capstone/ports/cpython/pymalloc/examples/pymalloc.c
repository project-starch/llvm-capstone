/* The real pymalloc API, with backing arenas supplied by the client. */
#include "port.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef PYMALLOC_POISONCAP
#include <sys/mman.h>
#endif
#include <string.h>

_Noreturn void pym_fail(unsigned code) {
  fprintf(stderr, "PYM example failed: %u\n", code);
  exit(1);
}
int main(void) {
  void *metadata = aligned_alloc(16384, PYM_META_BYTES);
#ifdef PYMALLOC_POISONCAP
  void *arena = mmap(NULL, PYM_ARENA_BYTES, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON | MAP_ALIGNED(14), -1, 0);
  if (arena == MAP_FAILED)
    return 4;
#else
  void *arena = aligned_alloc(16384, PYM_ARENA_BYTES);
#endif
  if (!metadata || !arena)
    return 1;
  pym_backing_init(metadata, arena);
#ifdef PYMALLOC_POISONCAP
  pym_lifetime_init(arena);
  pym_set_mode(1);
#endif
  pym_allocator_init();
  unsigned char *a = pym_malloc(48), *b = pym_calloc(1, 96);
  if (!a || !b || b[95])
    return 1;
  memset(a, 17, 48);
  a = pym_realloc(a, 160);
  if (!a || a[47] != 17)
    return 1;
  /* Realloc a capability-bearing object and check that its tag survives. */
  unsigned char **holder = pym_malloc(sizeof(*holder));
  if (!holder)
    return 1;
  *holder = b;
  holder = pym_realloc(holder, 1024);
  if (!holder || (*holder)[95])
    return 1;
  pym_free(holder);
  pym_free(a);
  pym_free(b);
#ifdef PYMALLOC_POISONCAP
  struct pym_header stats = {0};
  pym_backing_stats(&stats);
#endif
  free(metadata);
#ifdef PYMALLOC_POISONCAP
  munmap(arena, PYM_ARENA_BYTES);
#else
  free(arena);
#endif
  printf("ALLOCATOR_EXAMPLE cpython PASS pointer_bytes=%zu\n", sizeof(void *));
  return 0;
}
