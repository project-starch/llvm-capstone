#include "port.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef PYMALLOC_POISONCAP
#include <sys/mman.h>
#endif
#ifdef PYMALLOC_BORROW_LINEAR
/* The arena is LENT, not allocated: the system allocator hands over one linear
 * block and keeps the senior handle, which is what lets the nested adapter
 * revoke inside it and still lets the heap reclaim the whole thing. The helper
 * carves the alignment the adapter needs out of that block. */
#include "../../../../common/include/borrow-aligned-block.h"
#endif
_Noreturn void pym_fail(unsigned code) {
  fprintf(stderr, "PYM failed=%u\n", code);
#ifdef PYMALLOC_POISONCAP
  printf("PYM_REJECT code=%u\n", code);
  fflush(stdout);
#endif
  exit(1);
}
int main(int argc, char **argv) {
#ifdef PYMALLOC_CAPABILITY
  if (argc != 3 && argc != 4)
#else
  if (argc != 3)
#endif
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
#ifdef PYMALLOC_BORROW_LINEAR
  /* The adapter requires the arena pool-aligned and exactly PYM_ARENA_BYTES
   * long. The previous heap gave both by accident -- it rounded a request up to
   * a power of two and acquired each arena aligned to its own size -- and this
   * arm relied on that until the musl mallocng policy stopped providing it, at
   * which point all twenty cases refused to start. The alignment is made now,
   * and the refusal below stays: an arm that silently starts measuring
   * something else is worse than one that will not run. */
  capstone_cap_slot lent, head, tail;
  void *arena = NULL;
  /* 16 KiB, because pym_lifetime_init requires the arena on a pool boundary.
   * The heap's lend entry takes no alignment, so the helper makes one. */
  if (!capstone_borrow_aligned_block(PYM_ARENA_BYTES, 16384, &lent, &head, &tail)) {
    fprintf(stderr, "PYM failed=505 the lent arena is not a pool-aligned "
                    "linear region of %lu bytes\n", (unsigned long)PYM_ARENA_BYTES);
    return 4;
  }
#elif defined(PYMALLOC_POISONCAP)
  void *arena = mmap(NULL, PYM_ARENA_BYTES, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON | MAP_ALIGNED(14), -1, 0);
  if (arena == MAP_FAILED)
    return 4;
#else
  void *arena = aligned_alloc(16384, PYM_ARENA_BYTES);
#endif
#ifdef PYMALLOC_BORROW_LINEAR
  if (!metadata)
#else
  if (!metadata || !arena)
#endif
    return 4;
#ifdef PYMALLOC_CAPABILITY
  out.mode = 1;
  if (argc == 4) {
    char *end;
    unsigned long mode = strtoul(argv[3], &end, 10);
    if (!argv[3][0] || *end || mode > 1)
      return 2;
    out.mode = mode;
  }
#endif
  pym_backing_init(metadata, arena);
#ifdef PYMALLOC_BORROW_LINEAR
  /* The adapter takes the region as a pointer, because in a domain it arrives
   * in a register; capstone_cap_load moves it out of the slot so there is one
   * place holding it, which is what linear means. */
  pym_lifetime_init(capstone_cap_load(&lent));
  pym_set_mode(out.mode);
#elif defined(PYMALLOC_CAPABILITY)
  pym_lifetime_init(arena);
  pym_set_mode(out.mode);
#endif
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
#ifndef PYMALLOC_CAPABILITY
  /* backing.c keeps this counter only when IT owns the arena; with a
   * capability adapter the arena's decisions are the adapter's. */
  printf("PYM decisions=%016llx\n",
         (unsigned long long)pym_decision_checksum());
#endif
  free(metadata);
#ifdef PYMALLOC_BORROW_LINEAR
  /* The arena is the heap's; returning it is its senior handle's business, and
   * the process is about to end anyway. */
#elif defined(PYMALLOC_POISONCAP)
  munmap(arena, PYM_ARENA_BYTES);
#else
  free(arena);
#endif
  free(input);
  return 0;
}
