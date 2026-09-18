/* A bounded, reusable allocator for dependency metadata and backing buffers.
 * Blocks are reused only at the same rounded size. The reported memory curves
 * count requested pool payload bytes, not this heap's capacity or metadata.
 */
#include "metadata-allocator.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct block {
  struct block *next;
  size_t size;
};
static uint8_t *arena;
static size_t capacity, used;
static struct block *available;
void ff2_memory_init(void *base, size_t bytes) {
  arena = base;
  capacity = bytes;
  used = 0;
  available = NULL;
}
size_t ff2_metadata_used(void) { return used; }
void *av_malloc(size_t size) {
  if (size > capacity || size > SIZE_MAX - 63)
    return NULL;
  size = (size + 63) & ~(size_t)63;
  struct block **link = &available;
  while (*link) {
    struct block *b = *link;
    if (b->size == size) {
      *link = b->next;
      return (uint8_t *)b + 64;
    }
    link = &b->next;
  }
  if (capacity - used < 64 || size > capacity - used - 64)
    return NULL;
  struct block *b = (struct block *)(arena + used);
  b->size = size;
  used += 64 + size;
  return (uint8_t *)b + 64;
}
void av_free(void *p) {
  if (!p)
    return;
  struct block *b = (struct block *)((uint8_t *)p - 64);
  b->next = available;
  available = b;
}
void *av_mallocz(size_t size) {
  void *p = av_malloc(size);
  if (p)
    memset(p, 0, size);
  return p;
}
void av_freep(void *slot) {
  void **p = slot;
  av_free(*p);
  *p = NULL;
}
void *av_realloc(void *p, size_t size) {
  if (!size) {
    av_free(p);
    return NULL;
  }
  void *n = av_malloc(size);
  if (!n || !p)
    return n;
  size_t old = ((struct block *)((uint8_t *)p - 64))->size;
  memcpy(n, p, old < size ? old : size);
  av_free(p);
  return n;
}
void av_log(void *avcl, int level, const char *fmt, ...) {
  (void)avcl;
  (void)level;
  (void)fmt;
}
#ifdef FFPOOL_DOMAIN
_Noreturn void abort(void) { __builtin_trap(); }
#endif
