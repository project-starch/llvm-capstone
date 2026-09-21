/* The metadata heap. Adapter records, the page map and APR's allocator struct
 * live here and never inside a node, so revoking a node cannot revoke the
 * bookkeeping that describes it. Size-exact reuse, a 64-byte header per block,
 * no coalescing: the same shape as the pymalloc port's raw backing. */
#include "port.h"
#include <string.h>

struct block {
  struct block *next;
  size_t size;
};
static unsigned char *heap;
static size_t used;
static struct block *available;

void aprp_meta_init(void *metadata) {
  heap = metadata;
  used = 0;
  available = NULL;
}
void *aprp_meta_alloc(size_t n) {
  if (!heap)
    aprp_fail(510);
  if (n > APRP_META_BYTES - 64)
    return NULL;
  n = (n + 63) & ~(size_t)63;
  if (!n)
    n = 64;
  for (struct block **link = &available; *link; link = &(*link)->next)
    if ((*link)->size == n) {
      struct block *b = *link;
      *link = b->next;
      return (unsigned char *)b + 64;
    }
  if (used > APRP_META_BYTES - 64 - n)
    return NULL;
  struct block *b = (void *)(heap + used);
  b->size = n;
  used += n + 64;
  return (unsigned char *)b + 64;
}
void *aprp_meta_calloc(size_t k, size_t n) {
  if (n && k > SIZE_MAX / n)
    return NULL;
  void *p = aprp_meta_alloc(k * n);
  if (p)
    memset(p, 0, k * n);
  return p;
}
void aprp_meta_free(void *p) {
  if (!p)
    return;
  struct block *b = (void *)((unsigned char *)p - 64);
  b->next = available;
  available = b;
}
size_t aprp_meta_used(void) { return used; }
