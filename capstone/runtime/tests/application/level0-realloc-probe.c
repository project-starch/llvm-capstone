/* CPython reads 32 KiB, then retains a bytes object resized to the short read.
 * A shrinking realloc must release enough space to sustain that workload.
 * Also check payload capabilities, tail coalescing and regrowth after shrink. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef CAPSTONE_LEVEL0_TEST_NATIVE
#define malloc probe_malloc
#define free probe_free
#define calloc probe_calloc
#define realloc probe_realloc
#define __libc_malloc probe_libc_malloc
#define __libc_malloc_impl probe_libc_malloc_impl
#define __libc_calloc probe_libc_calloc
#define __libc_realloc probe_libc_realloc
#define __libc_free probe_libc_free
#define CAPSTONE_LEVEL0_ARENA_BYTES (1024 * 1024)
#define CAPSTONE_LEVEL0_STATS 1
#include "../../../ports/musl-capstone/runtime/level0.c"
void capstone_lock(volatile int *word) { (void)word; }
void capstone_unlock(volatile int *word) { (void)word; }
#endif

extern size_t __capstone_level0_in_use(void);
static void *items[4096];
static int marker = 73;

int main(void)
{
  size_t baseline = __capstone_level0_in_use();
  for (unsigned i = 0; i < 4096; ++i) {
    char *p = malloc(32768);
    if (!p) {
      fprintf(stderr, "short-read allocation %u failed\n", i);
      return 1;
    }
    *(int **)p = &marker;
    memset(p + sizeof(void *), (unsigned char)i, 64 - sizeof(void *));
    items[i] = realloc(p, 64);
    if (!items[i] || **(int **)items[i] != 73)
      return 2;
  }
  for (unsigned i = 0; i < 4096; ++i) {
    char *p = realloc(items[i], 80);
    if (!p || **(int **)p != 73)
      return 3;
    for (unsigned j = sizeof(void *); j < 64; ++j)
      if ((unsigned char)p[j] != (unsigned char)i)
        return 4;
    free(p);
  }
  if (__capstone_level0_in_use() != baseline)
    return 5;
  /* All released tails, moved blocks and freed neighbours must coalesce. */
  char *large = malloc(900 * 1024);
  if (!large)
    return 6;
  free(large);
  if (__capstone_level0_in_use() != baseline)
    return 7;
  puts("level0 realloc: 4096 short reads, capability payloads, regrowth and coalescing PASS");
  return 0;
}
