#include "corpus.h"

/* cache_create, fix 16a809e2a062. The crossed buffer is the object cache's own
 * freelist array -- one direct calloc of void* -- not an object the cache hands
 * out. That is what makes this row PLAIN rather than nested. */

MCH_CASE(3) {
  /* Case 3 -- cache_create's freelist array, fix 16a809e2a062. PLAIN HEAP: the
   * array of POINTERS was sized by the size of the object being cached.
   *
   * At the fix's parent:
   *
   *     void** ptr = calloc(initial_pool_size, bufsize);
   *
   * and the fix:
   *
   *     void** ptr = calloc(initial_pool_size, sizeof(void*));
   *
   * `ret->freetotal = initial_pool_size` promises 64 slots either way, and
   * do_cache_free stores up to ptr[63]. With a cached object smaller than a
   * pointer the array is short by the ratio, and that store is far outside it.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long initial_pool_size = 64;         /* upstream's constant */
  const unsigned long bufsize = 1;                    /* a one-byte cached object */
  const unsigned long bytes = fixed ? initial_pool_size * sizeof(void *)  /* 512 */
                                    : initial_pool_size * bufsize;        /* 64 */
  CHECK(bytes % 16 == 0, 911);  /* both arms land on a size class */

  void **ptr = calloc((size_t)bytes, 1);
  CHECK(ptr, 912);

  /* do_cache_free's last slot: ptr[freetotal - 1], i.e. byte offset
   * sizeof(void*) * 63 == 504 -- 440 bytes past the buggy arm's 64 bytes. The
   * probe touches the first byte past; the true distance is in defect_text. */
  const unsigned long store_off = sizeof(void *) * (initial_pool_size - 1);   /* 504 */
  const unsigned long touched = (store_off >= bytes) ? bytes : store_off;
  o->cap = bytes;
  o->touched = touched;
  o->crossed = store_off >= bytes;
  o->damage = o->crossed;
  if (o->crossed)
    write_probe((volatile unsigned char *)ptr + touched, 0xA5);   /* the labelled crossing */

  o->defect_text = "the freelist array of void* was sized by the CACHED OBJECT's size, so with a "
                   "one-byte object the 64 promised slots occupy 64 bytes and the last store lands "
                   "440 bytes past";
  o->fixed_text = "the fix sizes the array by sizeof(void*), so 64 slots really are 64 pointers";
  free(ptr);
}
