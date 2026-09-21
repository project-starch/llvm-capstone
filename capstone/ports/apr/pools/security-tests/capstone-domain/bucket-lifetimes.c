/* The bucket allocator's lifetime fixtures, one program, the fixture chosen
 * by the trace event's id. Each is the smallest sequence that reaches one of
 * the adapter's transitions, followed by the access that must FAULT in mode
 * 1 and COMPLETE in mode 0 -- or complete in both, for the controls -- at a
 * labelled instruction the runner compares the fault PC against.
 *
 *   0  live control: a freed small node is reissued at the same address,
 *      and the new holder reads its own byte
 *   1  a small node freed to the freelist, read through the old alias
 *   2  the same, written through the old alias after reissue: mode 0 corrupts
 *      the new holder silently, mode 1 faults at the write
 *   3  a large node -- a whole pool node -- freed, then read
 *   4  the allocator destroyed, its blocks back to APR, a small node read
 *   5  bounds: one byte past a small node's carved piece, both modes
 *   6  double free: the second apr_bucket_free reads through a revoked alias
 *      at the allocator's own labelled probe
 *
 * The domain reports nothing about itself: the runner reads the fault off
 * the monitor and the completion off the report. */
#include "port.h"
#include "apr_shim.h"
#include "apr_pools.h"
#include "apr_bucket_shim.h"
#include <string.h>

#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      aprp_fail(n);                                                            \
  } while (0)

static volatile unsigned char *held __attribute__((used));

__attribute__((noinline)) static unsigned read_probe(const volatile unsigned char *p) {
  unsigned long value;
  __asm__ volatile(".globl aprb_probe_read\naprb_probe_read:\nlbu %0, 0(%1)\n"
                   : "=r"(value)
                   : "r"(p)
                   : "memory");
  return value;
}
__attribute__((noinline)) static void write_probe(volatile unsigned char *p) {
  unsigned long value = 93;
  __asm__ volatile(".globl aprb_probe_write\naprb_probe_write:\nsb %0, 0(%1)\n" ::"r"(value),
                   "r"(p)
                   : "memory");
}
/* The fixture number and the three labelled sites: read, write, and the
 * allocator's own probe in apr_bucket_free. */
static void mark(unsigned id) {
  extern void aprb_probe_read(void), aprb_probe_write(void), aprb_free_probe(void);
  unsigned long code = 0xcf1b000000000000UL | id;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %1, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %2, x0\n"
                   ".insn r 0x5b, 0x1, 0x43, x0, %3, x0\n" ::"r"(code),
                   "r"(aprb_probe_read), "r"(aprb_probe_write), "r"(aprb_free_probe)
                   : "memory");
}

void aprp_replay(const struct aprp_header *input, struct aprp_header *out) {
  unsigned mode = out->mode;
  memset(out, 0, sizeof *out);
  out->mode = mode;
  out->magic = APRP_MAGIC;
  out->count = 1;
  const struct aprp_event *e = (const void *)(input + 1);
  unsigned test = e->id;
  CHECK(input->magic == APRP_MAGIC && input->count == 1 && test <= 6, 700);
  apr_pool_t *root = NULL;
  CHECK(apr_pool_create(&root, NULL) == APR_SUCCESS, 701);
  apr_bucket_alloc_t *ba = apr_bucket_alloc_create(root);
  CHECK(ba, 702);
  unsigned char *a = apr_bucket_alloc(64, ba);
  unsigned char *b = apr_bucket_alloc(64, ba);
  CHECK(a && b && a != b, 703);
  a[0] = 17;
  b[0] = 29;
  unsigned char *volatile stale = a;
  uintptr_t address = (uintptr_t)a;
  switch (test) {
  case 0: {
    apr_bucket_free(a);
    unsigned char *c = apr_bucket_alloc(64, ba);
    CHECK(c && (uintptr_t)c == address, 704); /* LIFO: the freed node comes back first */
    c[0] = 59;
    mark(test);
    CHECK(read_probe(c) == 59 && read_probe(b) == 29, 705);
    apr_bucket_free(c);
    apr_bucket_free(b);
    break;
  }
  case 1:
    apr_bucket_free(a);
    mark(test);
    (void)read_probe(stale);
    break;
  case 2: {
    apr_bucket_free(a);
    unsigned char *c = apr_bucket_alloc(64, ba);
    CHECK(c && (uintptr_t)c == address, 706);
    c[0] = 59;
    mark(test);
    write_probe(stale);
    CHECK(c[0] == 93, 707); /* mode 0: the new holder was written through the old alias */
    break;
  }
  case 3: {
    /* Above SMALL_NODE_SIZE: a whole APR node, freed straight back to the
     * pool allocator's free list. */
    unsigned char *big = apr_bucket_alloc(4096, ba);
    CHECK(big, 708);
    big[0] = 41;
    unsigned char *volatile stale_big = big;
    apr_bucket_free(big);
    mark(test);
    (void)read_probe(stale_big);
    break;
  }
  case 4:
    apr_bucket_alloc_destroy(ba);
    ba = NULL;
    mark(test);
    (void)read_probe(stale);
    break;
  case 5:
    /* The piece is SMALL_NODE_SIZE from the node header; the request's 64
     * bytes sit inside it, and the first byte past the PIECE is outside every
     * alias, in both modes. */
    mark(test);
    (void)read_probe(a + (APR_BUCKET_ALLOC_SIZE));
    break;
  case 6:
    apr_bucket_free(a);
    mark(test);
    apr_bucket_free(a); /* mode 1: faults at aprb_free_probe; mode 0: the adapter refuses (538) */
    break;
  }
  if (ba)
    apr_bucket_alloc_destroy(ba);
  apr_pool_destroy(root);
  out->completed = 1;
  aprp_stats(out);
}
