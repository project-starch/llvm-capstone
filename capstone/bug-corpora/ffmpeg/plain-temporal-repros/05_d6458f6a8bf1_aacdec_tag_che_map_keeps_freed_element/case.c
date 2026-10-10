#include "corpus.h"

/* libavcodec/aac/aacdec.c, che_configure, fix d6458f6a8bf1.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(5) {
  /* Case 5 -- che_configure's two tables, fix d6458f6a8bf1. The freed object is
   * a ChannelElement reachable from BOTH ac->che[][] (which owns it) and
   * ac->tag_che_map[][] (which caches it).
   *
   * At the fix's parent the free clears only the owner:
   *
   *     av_freep(&ac->che[type][id]);
   *
   * and the fix nulls the cache entries that name the same object first:
   *
   *     if (ac->tag_che_map[i][j] == ac->che[type][id])
   *         ac->tag_che_map[i][j] = NULL;
   *
   * Note av_freep IS the ender and this is still a use-after-free: the pointer
   * it clears is not the one followed later. */
  const unsigned long n = 48;
  unsigned char *che = malloc((size_t)n);         /* the owning slot */
  CHECK(che, 851);
  memset(che, 0x11, (size_t)n);
  volatile unsigned char *tag_che_map = che;      /* the cache slot: same object */

  if (fixed && tag_che_map == che)
    tag_che_map = NULL;                           /* the fix's cache walk */
  free(che);
  che = NULL;                                     /* av_freep clears the OWNER only */
  o->freed = 1;

  unsigned char *fresh = malloc((size_t)n);       /* tcache hands back the same chunk */
  CHECK(fresh, 852);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = tag_che_map ? read_probe(tag_che_map) : 0u;   /* the lookup's read */
  o->aliased = tag_che_map && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "tag_che_map still named the channel element after av_freep cleared only the "
                   "owning pointer, so the tag lookup returns storage that now belongs to another "
                   "object";
  o->fixed_text = "the fix nulls every tag_che_map entry naming the element before it is freed";
  free(fresh);
}
