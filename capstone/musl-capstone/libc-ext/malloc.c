/* malloc/free/realloc/calloc for libc-capstone.a.
 *
 * WHY THIS IS OURS AND NOT MUSL'S. Every file under musl's src/malloc and
 * src/malloc/mallocng fails to compile for capstone64: they carry static
 * asserts over sizeof(void*) that a 16-byte pointer violates. What survives in
 * the archive is only the shell -- free.o (needs __libc_free), realloc.o (needs
 * __libc_realloc) and lite_malloc.o (needs __mmap) -- three members whose bodies
 * are absent. build-musl-capstone.sh DELETES those three when it adds this file,
 * so a program cannot silently link the half that cannot work.
 *
 * WHY IT IS IN THE LIBC AND NOT IN EACH PROGRAM. It was in each program: the
 * Lua probe carried lua_probe_alloc.c, SQLite carries memsys5, the xlang shims
 * carry a third. An allocator per workload is one more thing to get wrong per
 * workload, and it makes "does a POSIX program link against our libc" depend on
 * the program bringing its own heap.
 *
 * THE CEILING, STATED PLAINLY: allocations are NOT individually bounded.
 * A block carved out of the static heap below carries the heap's bounds, so
 * capability spatial safety does not separate one allocation from the next here,
 * and nothing is revoked on free. That is deliberate for a bring-up allocator
 * and it is NOT the configuration any security number may be measured on: for
 * that, link xlang/common/revoke_arena_domain.c (revoke-on-free, per-allocation
 * SPLIT with its own revocation node), which needs a host-granted linear
 * capability to carve from. Bounding here would need the same machinery: once
 * the returned pointer is narrowed to its own allocation, free() can no longer
 * reach the header behind it, which is exactly why that allocator keeps a slot
 * table instead of an inline header.
 *
 * ALGORITHM: first fit over an address-ordered list, splitting on allocation and
 * coalescing forward on free. O(n) in live blocks per call, which is fine for
 * the interpreter-sized workloads this is for and is the next thing to change if
 * a workload allocates in the millions (segregated free lists, or the revoke
 * arena above).
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* 256 KiB, the size the Lua probe ran on. Deliberately modest: the heap is .bss,
 * and the domain loader rounds the whole image up to a power of two, so a
 * generous default would silently double a domain that never needed it.
 * Override with -DCAPSTONE_LIBC_HEAP_BYTES at build time. */
#ifndef CAPSTONE_LIBC_HEAP_BYTES
#define CAPSTONE_LIBC_HEAP_BYTES (256 * 1024)
#endif

/* 16, not 8: a capability is 16 bytes and must be 16-aligned, so any block that
 * might hold one has to start aligned. malloc's contract is alignment suitable
 * for every type, and on this target that type is a pointer. */
#define ALIGN 16

struct block {
  struct block *next; /* next block in ADDRESS order, or 0 */
  size_t size;        /* usable bytes after this header */
  int used;
  int pad;
};

static char heap[CAPSTONE_LIBC_HEAP_BYTES] __attribute__((aligned(ALIGN)));
static struct block *head;

static size_t round_up(size_t n) { return (n + (ALIGN - 1)) & ~(size_t)(ALIGN - 1); }

static void heap_init(void) {
  head = (struct block *)(void *)heap;
  head->next = 0;
  head->size = sizeof(heap) - sizeof(struct block);
  head->used = 0;
  head->pad = 0;
}

/* Split b so it serves exactly `want` bytes, if the remainder can hold a header
 * plus a minimum allocation. Without this a 4-byte request served from a large
 * free block keeps the whole block, and an interpreter that grows a table by
 * realloc walks the heap down in one direction and never comes back. */
static void split(struct block *b, size_t want) {
  if (b->size < want + sizeof(struct block) + ALIGN)
    return;
  struct block *rest = (struct block *)((char *)(b + 1) + want);
  rest->size = b->size - want - sizeof(struct block);
  rest->used = 0;
  rest->pad = 0;
  rest->next = b->next;
  b->next = rest;
  b->size = want;
}

void *malloc(size_t n) {
  if (!head)
    heap_init();
  if (n == 0)
    n = 1;
  size_t want = round_up(n);
  if (want < n) /* round_up overflowed: the request cannot be served */
    return 0;

  for (struct block *b = head; b; b = b->next) {
    if (b->used || b->size < want)
      continue;
    split(b, want);
    b->used = 1;
    return (void *)(b + 1);
  }
  return 0;
}

void free(void *p) {
  if (!p)
    return;
  struct block *b = (struct block *)p - 1;
  b->used = 0;
  /* Forward coalescing only. A backward merge would need a previous pointer or
   * a walk from head; the walk is what the next free does anyway, since every
   * free re-scans from the block it just released. */
  while (b->next && !b->next->used) {
    b->size += sizeof(struct block) + b->next->size;
    b->next = b->next->next;
  }
}

void *realloc(void *p, size_t n) {
  if (!p)
    return malloc(n);
  if (n == 0) {
    free(p);
    return 0;
  }
  struct block *b = (struct block *)p - 1;
  size_t want = round_up(n);

  /* Grow in place when the next block is free and adjacent: the interpreter
   * case this exists for is a table doubling repeatedly at the top of the heap,
   * where copying every time is the whole cost. */
  while (b->size < want && b->next && !b->next->used) {
    b->size += sizeof(struct block) + b->next->size;
    b->next = b->next->next;
  }
  if (b->size >= want) {
    split(b, want);
    return p;
  }

  void *fresh = malloc(n);
  if (!fresh)
    return 0; /* the original stays valid, as realloc requires */
  /* Copied by hand rather than with memcpy: the 9 word-at-a-time src/string
   * files do not compile for this target ((uintptr_t)s % ALIGN), so memcpy is
   * absent from the archive and calling it here would make the libc's allocator
   * depend on each program supplying one. memset below IS in the archive. */
  const char *from = p;
  char *to = fresh;
  for (size_t i = 0; i < b->size; i++)
    to[i] = from[i];
  free(p);
  return fresh;
}

void *calloc(size_t count, size_t size) {
  if (count && size > (size_t)-1 / count)
    return 0; /* overflow: the multiply would wrap and under-allocate */
  size_t total = count * size;
  void *p = malloc(total);
  if (p)
    memset(p, 0, total);
  return p;
}
