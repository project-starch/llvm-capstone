#include "corpus.h"

/* restart.c, restart_get_kv, fix 0d4901071c74.
 * The object is ONE direct malloc-family allocation -- neither slabs.c nor
 * cache.c is in the path -- which is what puts this row in the plain-temporal
 * corpus rather than ../allocator-repros. */

MCT_CASE(0) {
  /* Case 0 -- restart_get_kv's line buffer, fix 0d4901071c74. The freed object
   * is the getline() buffer; what is left holding its address is c->line,
   * because the entry free does not clear it and three of the four return paths
   * never republish.
   *
   * At the fix's parent:
   *
   *     if (c->line != NULL) {
   *         free(c->line);
   *     }
   *
   * and the fix clears the field beside the free:
   *
   *         free(c->line);
   *         c->line = NULL;
   */
  const unsigned long n = 48;
  unsigned char *line = malloc((size_t)n);        /* getline()'s allocation */
  CHECK(line, 801);
  memset(line, 0x11, (size_t)n);

  volatile unsigned char *c_line = line;          /* the field */
  free(line);                                     /* the entry free */
  o->freed = 1;
  if (fixed)
    c_line = NULL;                                /* the fix's c->line = NULL */
  /* The buggy arm returns RESTART_DONE here, so nothing republishes the field. */

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 802);
  CHECK_REUSE(fresh == line, 803);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = c_line ? read_probe(c_line) : 0u; /* the NEXT call's entry free */
  o->aliased = c_line && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "the entry free did not clear c->line, and three of the four return paths never "
                   "republish it, so the next call releases storage that now belongs to another "
                   "object";
  o->fixed_text = "the fix clears c->line beside the free, so re-entry is safe however the function "
                  "returned";
  free(fresh);
}
