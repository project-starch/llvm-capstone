#ifndef PYMALLOC_PORT_H
#define PYMALLOC_PORT_H
#include <stddef.h>
#include <stdint.h>
#define PYM_ARENA_BYTES (64UL * 1024 * 1024)
#define PYM_META_BYTES (16UL * 1024 * 1024)
#define PYM_FILE_BYTES (8UL * 1024 * 1024)
#define PYM_MAX_OBJECTS 65536
#define PYM_MAGIC UINT64_C(0x31594c50524d5950)
enum { PYM_ALLOC = 1, PYM_CALLOC, PYM_REALLOC, PYM_FREE, PYM_END };
struct pym_event {
  uint64_t op, id, size, value;
};
struct pym_header {
  uint64_t magic, count, mode, status, completed, allocations, frees,
      reallocations, arenas, arena_frees, metadata, checksum;
};
_Noreturn void pym_fail(unsigned code);
void pym_backing_init(void *metadata, void *arena);
void *pym_raw_malloc(size_t);
void *pym_raw_calloc(size_t, size_t);
void *pym_raw_realloc(void *, size_t);
void pym_raw_free(void *);
void *pym_arena_alloc(void *, size_t);
void pym_arena_free(void *, void *, size_t);
void *pym_arena_pointer(uintptr_t);
void *pym_pool_pointer(const void *);
void pym_backing_stats(struct pym_header *);
uintptr_t pym_arena_address(void *);
void *pym_pool_create(uintptr_t, size_t);
void pym_pool_reclass(void *, size_t, size_t);
void *pym_block_pointer(void *, size_t);
void *pym_issue(void *, size_t);
void *pym_release(void *);
void *pym_resize(void *, size_t);
size_t pym_requested(void *);
void pym_validate(void *);
void *pym_user_raw_malloc(size_t);
void pym_user_raw_free(void *);
void *pym_user_raw_realloc(void *, size_t);
void pym_lifetime_init(void *);
void pym_set_mode(unsigned);
void pym_observe(unsigned, void *, size_t);
uint64_t pym_decision_checksum(void);
void pym_allocator_init(void);
void *pym_malloc(size_t);
void *pym_calloc(size_t, size_t);
void *pym_realloc(void *, size_t);
void pym_free(void *);
void pym_replay(const struct pym_header *, struct pym_header *, void *scratch);
#endif
