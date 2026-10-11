/* Case 0: 7af02b0c87 -- a text multiget's read buffer is copied after it
 * went back to the thread's rbuf cache
 *
 * Shape: stale object pointer / cache.c reuse / read through the dead pointer
 * Consumer: memcached.c, rbuf_switch_to_malloc()
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

#define READ_BUFFER_SIZE 16384 /* memcached.h:80 */
/* rbuf_switch_to_malloc takes the larger buffer from malloc; a domain has no
 * malloc, and the defect is not about that buffer, so it is static here. */
static char switched[2 * READ_BUFFER_SIZE];

MC_CASE(0) {
  o->defect_text = "the command bytes copied out of the freed buffer are the free list's link";
  o->fixed_text = "the copy happened first; the buffer went back afterwards";

  cache_t *rbuf_cache = cache_create("rbuf", READ_BUFFER_SIZE, sizeof(char *));
  CHECK(rbuf_cache, 710);
  /* Another connection's buffer, taken first and given back once this one
   * holds its own, so it sits on the thread's free list when the defect
   * frees: the link written into the freed buffer is then a real pointer, not
   * NULL. Checked, because a LIFO cache would otherwise hand the same buffer
   * to both and the free list would be empty at the moment that matters. */
  char *previous = cache_alloc(rbuf_cache);
  CHECK(previous, 711);

  /* This connection: a multiget too large for one read buffer. */
  char *rbuf = cache_alloc(rbuf_cache);
  CHECK(rbuf && rbuf != previous, 712);
  cache_free(rbuf_cache, previous);
  static const char command[] = "get key0001 key0002 key0003 key0004 key0005 key0006 key0007\r\n";
  size_t rbytes = sizeof command - 1;
  memcpy(rbuf, command, rbytes);
  char *rcurr = rbuf;

  /* rbuf_switch_to_malloc, as it stood before the fix: give the cache buffer
   * back, THEN copy the unparsed command out of it. The buffer's address is
   * taken now, while the pointer is live; in the protected arm it is dead
   * after the free and cannot even be compared. */
  uintptr_t buffer = (uintptr_t)rbuf;
  held = (volatile unsigned char *)rcurr;
  if (!fixed) {
    cache_free(rbuf_cache, rbuf);
    mark(0);
    (void)read_probe(held);
    memcpy(switched, rcurr, rbytes);
    o->accessed_through_stale = 1;
  } else {
    memcpy(switched, rcurr, rbytes); /* upstream 7af02b0c87: copy first */
    cache_free(rbuf_cache, rbuf);
  }
  rcurr = rbuf = switched;
  o->damage = memcmp(switched, command, rbytes) != 0;

  /* The next connection on this thread takes a read buffer: the one just
   * freed, because the cache is LIFO. Recorded as the mechanism, in both arms. */
  char *next = cache_alloc(rbuf_cache);
  CHECK(next, 713);
  o->unit_reissued = (uintptr_t)next == buffer;
  cache_free(rbuf_cache, next);
  cache_destroy(rbuf_cache);
}
