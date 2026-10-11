#include "corpus.h"

/* mcp_key_hash_filter_tag, fix 212c3820c7bb. The key buffer is ONE direct
 * allocation; neither slabs.c nor cache.c is involved. The function is static
 * and pure, so the reduction needs no lua. */

MCH_CASE(6) {
  /* Case 6 -- the proxy's key hash tag filter, fix 212c3820c7bb. PLAIN HEAP:
   * the search for the CLOSING tag started at the OPENING tag's own position.
   *
   * At the fix's parent:
   *
   *     const char *t2 = memchr(t1, conf[1], remain);
   *     if (t2) {
   *         *newlen = t2 - t1 - 1;
   *         return t1+1;
   *     }
   *
   * and the fix starts one past and shortens the span to match:
   *
   *     const char *t2 = memchr(t1+1, conf[1], remain-1);
   *
   * The function's own comment permits a two-character conf whose characters are
   * EQUAL, such as "$$". Then memchr finds conf[1] at t1 itself, so t2 == t1 and
   * `*newlen = t2 - t1 - 1` underflows to SIZE_MAX. The caller then runs
   * `p->key_hasher(key, len, p->hash_seed)` with that length.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long klen = 16;            /* a size class: no slack to absorb the crossing */
  unsigned char *key = calloc((size_t)klen, 1);
  CHECK(key, 941);
  memcpy(key, "ab$cdefghijklmno", (size_t)klen);

  const unsigned long t1 = 2;               /* the opening tag's position: key[2] == '$' */
  CHECK(key[t1] == '$', 942);
  const unsigned long remain = klen - t1;

  /* conf is "$$": the two tag characters are equal, which is what makes the
   * buggy search find the opening tag as if it were the closing one. */
  unsigned long newlen;
  int found;
  if (fixed) {
    const void *t2 = memchr(key + t1 + 1, '$', (size_t)(remain - 1));
    found = t2 != NULL;                     /* no second '$': the filter declines */
    newlen = found ? (unsigned long)((const unsigned char *)t2 - (key + t1) - 1) : 0;
  } else {
    const void *t2 = memchr(key + t1, '$', (size_t)remain);
    found = t2 != NULL;                     /* finds key[t1] itself */
    newlen = (unsigned long)((const unsigned char *)t2 - (key + t1) - 1);  /* SIZE_MAX */
  }

  /* The hasher would read `newlen` bytes from key + t1 + 1. The probe touches
   * the first byte outside the allocation; the unreduced read is unbounded. */
  const unsigned long base = t1 + 1;
  const unsigned long reach = (found && newlen > klen) ? klen : base + newlen;
  o->cap = klen;
  o->touched = reach;
  o->crossed = found && reach >= klen;
  o->damage = o->crossed;
  if (o->crossed)
    (void)read_probe(key + klen);           /* the labelled crossing */

  o->defect_text = "the closing-tag search started at the opening tag, so an equal-character tag "
                   "conf made t2 == t1 and the hashed length underflowed to SIZE_MAX -- an "
                   "unbounded read";
  o->fixed_text = "the fix searches from t1+1 over remain-1, so an absent closing tag declines "
                  "instead of underflowing";
  free(key);
}
