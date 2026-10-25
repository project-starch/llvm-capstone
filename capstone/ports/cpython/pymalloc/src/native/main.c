#include "port.h"
#include <stdio.h>
#include <stdlib.h>
_Noreturn void pym_fail(unsigned code) {
  fprintf(stderr, "PYM failed=%u\n", code);
  exit(1);
}
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  FILE *f = fopen(argv[1], "rb");
  struct pym_header *input = malloc(PYM_FILE_BYTES), out = {0};
  if (!f || !input)
    return 2;
  size_t bytes = fread(input, 1, PYM_FILE_BYTES, f);
  if (ferror(f) || fgetc(f) != EOF || bytes < sizeof *input ||
      input->count >
          (PYM_FILE_BYTES - sizeof *input) / sizeof(struct pym_event) ||
      bytes != sizeof *input + input->count * sizeof(struct pym_event))
    return 3;
  fclose(f);
  void *metadata = aligned_alloc(16384, PYM_META_BYTES);
  void *arena = aligned_alloc(16384, PYM_ARENA_BYTES);
  if (!metadata || !arena)
    return 4;
  pym_backing_init(metadata, arena);
  pym_allocator_init();
  void *scratch = pym_raw_calloc(PYM_MAX_OBJECTS, 64);
  if (!scratch)
    return 4;
  pym_replay(input, &out, scratch);
  f = fopen(argv[2], "wb");
  if (!f || fwrite(&out, sizeof out, 1, f) != 1 || fclose(f))
    return 5;
  printf("PYM completed=%llu alloc=%llu free=%llu realloc=%llu arenas=%llu "
         "released=%llu\n",
         (unsigned long long)out.completed, (unsigned long long)out.allocations,
         (unsigned long long)out.frees, (unsigned long long)out.reallocations,
         (unsigned long long)out.arenas, (unsigned long long)out.arena_frees);
  printf("PYM decisions=%016llx\n",
         (unsigned long long)pym_decision_checksum());
  free(metadata);
  free(arena);
  free(input);
  return 0;
}
