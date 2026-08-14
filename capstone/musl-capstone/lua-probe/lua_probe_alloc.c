/* A small first-fit allocator for the probe.
 *
 * DELIBERATELY NOT xlang/common/revoke_arena_domain.c. That one answers a
 * different question -- "is the free revoked?" -- and needs a LINEAR capability
 * granted by the host to carve from. This probe asks only "does reference Lua
 * run against musl in a domain", so mixing in the revoke machinery would add a
 * second variable to a first measurement. The revoke arena is the next step,
 * not this one.
 *
 * It is also why musl's own malloc is not used: every file under src/malloc
 * fails to compile for this target (static asserts over sizeof(void*), violated
 * by 16-byte pointers), so nothing in libc-capstone.a defines malloc. Defining
 * malloc/free/realloc/calloc here additionally stops musl's free.o and realloc.o
 * being pulled from the archive, which is what removes the undefined
 * __libc_free and __libc_realloc.
 *
 * First fit with a free list and forward coalescing. No splitting of a reused
 * block: a request served from a larger free block keeps the larger size, which
 * wastes memory on a workload with many differently-sized reallocs. Acceptable
 * for a probe that runs one small chunk; the upgrade path is the revoke arena
 * above, not a better allocator here.
 */
#include <stddef.h>
#include <stdint.h>

#ifndef LUA_PROBE_HEAP_BYTES
#define LUA_PROBE_HEAP_BYTES (256 * 1024)
#endif

struct block {
  struct block *next;   /* next block in address order */
  size_t size;          /* usable bytes after this header */
  int used;
  int pad;
};

static char heap[LUA_PROBE_HEAP_BYTES] __attribute__((aligned(16)));
static struct block *head;

static void heap_init(void) {
  head = (struct block *)(void *)heap;
  head->next = 0;
  head->size = sizeof(heap) - sizeof(struct block);
  head->used = 0;
}

static void coalesce(void) {
  for (struct block *b = head; b; b = b->next)
    while (!b->used && b->next && !b->next->used) {
      b->size += sizeof(struct block) + b->next->size;
      b->next = b->next->next;
    }
}

void *malloc(size_t n) {
  if (!head)
    heap_init();
  if (!n)
    n = 1;
  n = (n + 15u) & ~(size_t)15u;

  for (int pass = 0; pass < 2; pass++) {
    for (struct block *b = head; b; b = b->next) {
      if (b->used || b->size < n)
        continue;
      /* Split only when the tail is worth a header. */
      if (b->size >= n + sizeof(struct block) + 16) {
        struct block *tail = (struct block *)((char *)(b + 1) + n);
        tail->next = b->next;
        tail->size = b->size - n - sizeof(struct block);
        tail->used = 0;
        b->next = tail;
        b->size = n;
      }
      b->used = 1;
      return b + 1;
    }
    coalesce();   /* only worth doing once, before giving up */
  }
  return 0;
}

void free(void *p) {
  if (!p)
    return;
  struct block *b = (struct block *)p - 1;
  b->used = 0;
}

void *realloc(void *p, size_t n) {
  if (!p)
    return malloc(n);
  if (!n) {
    free(p);
    return 0;
  }
  struct block *b = (struct block *)p - 1;
  if (b->size >= n)
    return p;
  void *fresh = malloc(n);
  if (!fresh)
    return 0;
  const char *src = (const char *)p;
  char *dst = (char *)fresh;
  for (size_t i = 0; i < b->size; i++)
    dst[i] = src[i];
  free(p);
  return fresh;
}

void *calloc(size_t count, size_t size) {
  size_t total = count * size;
  if (count && total / count != size)   /* overflow check, cheap and required */
    return 0;
  char *p = (char *)malloc(total);
  if (p)
    for (size_t i = 0; i < total; i++)
      p[i] = 0;
  return p;
}

/* Two more musl internals whose own source files do not compile for this target
   (src/string/strchrnul.c is one of the nine word-at-a-time string files;
   src/thread/__lock.c is one of the thread files). A domain is single-threaded,
   so the locks are genuinely nothing. */
char *__strchrnul(const char *s, int c) {
  for (; *s && *s != (char)c; s++)
    ;
  return (char *)s;
}
void __lock(volatile int *l) { (void)l; }
void __unlock(volatile int *l) { (void)l; }
