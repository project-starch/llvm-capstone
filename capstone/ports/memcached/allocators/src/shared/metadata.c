/* The metadata heap. Adapter records, the page map, the slab lists and
 * cache.c's control blocks live here and never inside a page, so revoking a
 * chunk cannot revoke the bookkeeping that describes it. Size-exact reuse, a
 * 64-byte header per block, no coalescing: the same shape as the pymalloc and
 * APR ports' metadata heaps. */
#include "port.h"
#include <string.h>

struct block {
  struct block *next;
  size_t size;
};
static unsigned char *heap;
static size_t used;
static struct block *available;

void mcp_meta_init(void *metadata) {
  heap = metadata;
  used = 0;
  available = NULL;
}
void *mcp_meta_alloc(size_t n) {
  if (!heap)
    mcp_fail(510);
  if (n > MCP_META_BYTES - 64)
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
  if (used > MCP_META_BYTES - 64 - n)
    return NULL;
  struct block *b = (void *)(heap + used);
  b->size = n;
  used += n + 64;
  return (unsigned char *)b + 64;
}
void *mcp_meta_calloc(size_t k, size_t n) {
  if (n && k > SIZE_MAX / n)
    return NULL;
  void *p = mcp_meta_alloc(k * n);
  if (p)
    memset(p, 0, k * n);
  return p;
}
void mcp_meta_free(void *p) {
  if (!p)
    return;
  struct block *b = (void *)((unsigned char *)p - 64);
  b->next = available;
  available = b;
}
char *mcp_meta_strdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *d = mcp_meta_alloc(n);
  if (d)
    memcpy(d, s, n);
  return d;
}
/* do_grow_slab_list's realloc. Element-wise on purpose: the list holds page
 * aliases, capabilities in a domain, and a byte copy would not carry them. */
void **mcp_meta_grow_pointers(void **old, size_t count, size_t new_count) {
  if (new_count < count)
    mcp_fail(511);
  void **grown = mcp_meta_alloc(new_count * sizeof *grown);
  if (!grown)
    return NULL;
  for (size_t i = 0; i < count; ++i)
    grown[i] = old[i];
  for (size_t i = count; i < new_count; ++i)
    grown[i] = NULL;
  mcp_meta_free(old);
  return grown;
}
size_t mcp_meta_used(void) { return used; }
