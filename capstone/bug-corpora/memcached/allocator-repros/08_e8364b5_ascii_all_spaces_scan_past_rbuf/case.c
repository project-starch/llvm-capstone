/* Case 8: e8364b5 -- the ASCII large-multiget path skips leading spaces with an
 * unbounded scan, so a read buffer that is entirely spaces walks the cursor off
 * the end of the cache object and on into the next one.
 *
 * Shape: an unbounded scan runs past a cache object into the next object of the
 *        same cache
 * Consumer: proto_text.c, try_read_command_ascii's large-multiget branch
 * SPATIAL -- no lifetime ends; the buffer is alive and the cursor simply leaves
 * it. The corpus's fourth spatial case, and the first whose object comes from
 * cache.c rather than from slabs.c.
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

#define READ_BUFFER_SIZE 16384 /* memcached.h:80, as case 0 also declares it */

MC_CASE(8) {
  o->defect_text = "the leading-space skip had no end, so a read buffer of all spaces "
                   "walked the cursor past the cache object into the next one";
  o->fixed_text = "the fix's `ptr != end` bound stops the scan inside the buffer";

  /* The read buffers are cache.c objects, exactly as memcached.c:408 takes them
   * (`c->rbuf = do_cache_alloc(c->thread->rbuf_cache)`), which is what makes
   * this a NESTED spatial row: the crossing leaves the object, not the page. */
  cache_t *rbuf_cache = cache_create("rbuf", READ_BUFFER_SIZE, sizeof(char *));
  CHECK(rbuf_cache, 780);

  /* Two objects of the same cache. Their order is CHECKED rather than assumed:
   * the free list does not promise ascending addresses, and the lower object is
   * the one the scan runs out of. Case 5 records the same trap -- its first
   * version asserted the wrong order and the control refused the run. */
  char *first = cache_alloc(rbuf_cache);
  CHECK(first, 781);
  char *second = cache_alloc(rbuf_cache);
  CHECK(second && second != first, 782);
  char *rbuf = first < second ? first : second;
  char *successor = first < second ? second : first;
  CHECK((uintptr_t)successor > (uintptr_t)rbuf, 783);

  /* The reachable input: a buffer with no '\n' in the first few k, which is the
   * branch's own premise ("This _has_ to be a large multiget"), and nothing but
   * spaces in it. The successor is filled with spaces too -- which is what
   * makes the scan CONTINUE rather than stop at the first byte past the end,
   * and is why the defect is an overrun rather than a single stray read. */
  memset(rbuf, ' ', READ_BUFFER_SIZE);
  memset(successor, ' ', READ_BUFFER_SIZE);
  size_t rbytes = READ_BUFFER_SIZE;

  /* try_read_command_ascii at the pin, :370-372:
   *
   *     char *ptr = c->rcurr;
   *     while (*ptr == ' ') {   // ignore leading whitespaces
   *         ++ptr;
   *     }
   *
   * and the fix, which gives the scan an end derived from the bytes actually
   * read:
   *
   *     char *end = c->rcurr + c->rbytes-6;
   *     while (*ptr == ' ' && ptr != end) {
   */
  char *rcurr = rbuf;
  char *end = rcurr + rbytes - 6;      /* the fix's bound */
  char *ptr = rcurr;
  unsigned long steps = 0;
  while (*(volatile char *)ptr == ' ') {
    if (fixed && ptr == end)
      break;
    ++ptr;
    if (++steps > READ_BUFFER_SIZE + 32) /* the reduction's own stop */
      break;
  }

  /* Both halves of the claim, asserted rather than assumed: at the pin the
   * cursor ends PAST the object, and under the fix it does not. A reduction
   * whose arithmetic missed fails here instead of reporting a verdict. */
  if (fixed) {
    CHECK(ptr <= rcurr + rbytes, 784);
  } else {
    CHECK(ptr > rcurr + rbytes, 785);
  }
  /* And the crossing stays inside the PAGE the cache carved both objects from,
   * which is what makes a page-granular bound blind to it. */
  CHECK((uintptr_t)ptr <= (uintptr_t)successor + READ_BUFFER_SIZE, 786);

  o->unit_reissued = 0;                /* nothing is freed or reissued here */
  held = (unsigned char *)rcurr + rbytes;  /* set while live, as the seam requires */
  mark(8); /* LAST thing before the access: its presence is the setup's evidence */
  (void)read_probe((const volatile unsigned char *)held);

  o->accessed_through_stale = !fixed;  /* the crossing happened */
  o->damage = !fixed;
  cache_free(rbuf_cache, second);
  cache_free(rbuf_cache, first);
}
