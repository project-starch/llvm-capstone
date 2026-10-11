#include "corpus.h"

/* wiretap/peak-trc.c, peak_trc_open's not-mine path, fix 7dcf69480de8.
 * The object is ONE direct g_malloc-family allocation -- no wmem -- which is
 * what puts this row in the plain-temporal corpus rather than ../wmem-repros. */

WST_CASE(1) {
  /* Case 1 -- peak_trc_open's not-mine path, fix 7dcf69480de8. The freed object
   * is the reader state; the pointer left holding its address is the caller's
   * `trc_state`, because clean_trc_state frees the struct it is handed.
   *
   * At the fix's parent:
   *
   *     clean_trc_state(trc_state);
   *     g_free(trc_state);
   *
   * and the fix deletes the caller's free. */
  const unsigned long n = 48;
  unsigned char *trc_state = malloc((size_t)n);
  CHECK(trc_state, 811);
  memset(trc_state, 0x11, (size_t)n);

  volatile unsigned char *stale = trc_state;
  free(trc_state);                 /* clean_trc_state's own g_free(state) */
  o->freed = 1;
  if (fixed)
    stale = NULL;

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 812);
  CHECK_REUSE(fresh == trc_state, 813);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = stale ? read_probe(stale) : 0u;
  o->aliased = stale && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "clean_trc_state frees the state struct it is handed, so the caller's following "
                   "g_free releases storage that now belongs to another object";
  o->fixed_text = "the fix deletes the caller's free, leaving the helper the single owner";
  free(fresh);
}
