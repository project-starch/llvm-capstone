/* Successful API controls complement the faulting lease probes. */
#include "port.h"
#include <cheri/cheric.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define CHECK(x, n) do { if (!(x)) pym_fail(n); } while (0)
_Noreturn void pym_fail(unsigned code) {
  fprintf(stderr, "PYM_CHECK failed=%u\n", code);
  exit(1);
}
static void sizes(void) {
  for (size_t n = 0; n <= 512; n += 16) {
    unsigned char *p = pym_calloc(1, n);
    CHECK(p && cheri_gettag(p) && cheri_getlen(p) >= (n ? n : 1), 801);
    CHECK(!(cheri_getperm(p) & (CHERI_PERM_POISON | CHERI_PERM_SW_VMEM)), 802);
    for (size_t i = 0; i < n; ++i)
      CHECK(!p[i], 803);
    pym_free(p);
  }
  const size_t raw[] = {513, 4096, 65537, 1048576};
  for (size_t i = 0; i < sizeof raw / sizeof *raw; ++i) {
    unsigned char *p = pym_malloc(raw[i]);
    CHECK(p && cheri_getlen(p) >= raw[i], 804);
    p[0] = 13;
    p[raw[i] - 1] = 37;
    CHECK(p[0] == 13 && p[raw[i] - 1] == 37, 805);
    pym_free(p);
  }
  CHECK(!pym_calloc(SIZE_MAX, 16), 806);
}
static void resize(void) {
  unsigned char *sibling = pym_malloc(32);
  CHECK(sibling, 807);
  sibling[0] = 53;
  const size_t initial[] = {64, 1024};
  for (size_t i = 0; i < 2; ++i) {
    void **holder = pym_malloc(initial[i]);
    CHECK(holder, 808);
    holder[0] = sibling;
    uintptr_t address = (uintptr_t)holder;
    holder = pym_realloc(holder, initial[i] - 1);
    CHECK(holder && (uintptr_t)holder == address && cheri_gettag(holder[0]) &&
          ((unsigned char *)holder[0])[0] == 53, 809);
    void **moved = pym_realloc(holder, 4096);
    CHECK(moved && cheri_gettag(moved[0]) && ((unsigned char *)moved[0])[0] == 53, 810);
    CHECK(!pym_realloc(moved, PYM_ARENA_BYTES) &&
          ((unsigned char *)moved[0])[0] == 53, 811);
    pym_free(moved);
  }
  pym_free(sibling);
}
static void snapshot_failure(void) {
  unsigned char *p = pym_malloc(64);
  CHECK(p, 812);
  memset(p, 61, 64);
  /* Exhaust only adapter metadata; the existing object's storage stays live. */
  while (pym_raw_malloc(65536)) {}
  while (pym_raw_malloc(64)) {}
  CHECK(!pym_realloc(p, 63), 813);
  for (size_t i = 0; i < 64; ++i)
    CHECK(p[i] == 61, 814);
  pym_free(p);
}
static void unwritten_reuse(void) {
  /* A retired poison capability must not revoke a later unwritten allocation
   * at the same address when another object's free triggers a kernel sweep. */
  const size_t sizes[] = {0, 1094};
  for (size_t i = 0; i < 2; ++i) {
    void *first = pym_malloc(sizes[i]);
    CHECK(first, 822);
    uintptr_t address = (uintptr_t)first;
    pym_free(first);
    void *fresh = pym_malloc(sizes[i]);
    CHECK(fresh && (uintptr_t)fresh == address, 823);
    void *other = pym_malloc(2048);
    CHECK(other, 824);
    pym_free(other);
    CHECK(cheri_gettag(fresh), 825);
    pym_free(fresh);
  }
}
static void arena_turnover(void) {
  enum { COUNT = 2300 }; /* More than one arena of 512-byte blocks. */
  unsigned char **objects = calloc(COUNT, sizeof *objects);
  CHECK(objects, 815);
  for (size_t i = 0; i < COUNT; ++i) {
    objects[i] = pym_malloc(512);
    CHECK(objects[i], 816);
    objects[i][0] = i % 251;
  }
  for (size_t i = 0; i < COUNT; ++i) {
    CHECK(objects[i][0] == i % 251, 817);
    pym_free(objects[i]);
  }
  struct pym_header stats = {0};
  pym_backing_stats(&stats);
  CHECK(stats.arenas > 1 && stats.arena_frees > 0, 818);
  unsigned char *p = pym_malloc(16);
  CHECK(p, 819);
  p[0] = 73;
  CHECK(p[0] == 73, 820);
  pym_free(p);
  free(objects);
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  void *metadata = aligned_alloc(16384, PYM_META_BYTES);
  void *arena = mmap(NULL, PYM_ARENA_BYTES, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON | MAP_ALIGNED(14), -1, 0);
  CHECK(metadata && arena != MAP_FAILED, 821);
  pym_backing_init(metadata, arena);
  pym_lifetime_init(arena);
  pym_set_mode(1);
  pym_allocator_init();
  if (!strcmp(argv[1], "sizes"))
    sizes();
  else if (!strcmp(argv[1], "realloc"))
    resize();
  else if (!strcmp(argv[1], "snapshot-failure"))
    snapshot_failure();
  else if (!strcmp(argv[1], "unwritten-reuse"))
    unwritten_reuse();
  else if (!strcmp(argv[1], "arena-turnover"))
    arena_turnover();
  else
    return 2;
  struct pym_header stats = {0};
  pym_backing_stats(&stats);
  printf("PYM_CHECK %s PASS pointer_bytes=%zu\n", argv[1], sizeof(void *));
  return 0;
}
