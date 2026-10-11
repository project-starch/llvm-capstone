#include "corpus.h"

/* wiretap/wtap.c, wtap_close's interface teardown, fix fb46cda19602.
 * The object is ONE direct g_malloc-family allocation -- no wmem -- which is
 * what puts this row in the plain-temporal corpus rather than ../wmem-repros. */

WST_CASE(5) {
  /* Case 5 -- wtap_close's shared loop index, fix fb46cda19602. The freed
   * objects are the per-interface description strings; what makes them be freed
   * twice is that the INNER loop reuses the OUTER loop's index, rewinding it.
   *
   * At the fix's parent:
   *
   *     gint i;
   *     for(i = 0; i < (gint)wth->number_of_interfaces; i++) {
   *         ...
   *         for(i = 0; i < (gint)wtapng_if_descr->num_stat_entries; i++) {
   *
   * and the fix gives the inner loop its own index:
   *
   *     gint i, j;
   *         for(j = 0; j < (gint)wtapng_if_descr->num_stat_entries; j++) {
   */
  const unsigned long n = 48;
  unsigned char *opt_comment = malloc((size_t)n);   /* one description string */
  CHECK(opt_comment, 851);
  memset(opt_comment, 0x11, (size_t)n);

  /* The outer pass: free the string. It is NOT set to NULL, which is what makes
   * the rewind a second release rather than a no-op. */
  volatile unsigned char *stale = opt_comment;
  free(opt_comment);
  o->freed = 1;

  /* The inner loop. In the buggy arm it shares the outer index, so after it the
   * outer cursor is back at 0 and the same field is visited again. */
  unsigned long i = 0;
  const unsigned long stat_entries = 1;
  if (fixed) {
    for (unsigned long j = 0; j < stat_entries; j++) { /* its own index */ }
    stale = NULL;                                  /* the outer cursor advances past the field */
  } else {
    for (i = 0; i < stat_entries; i++) { /* reuses the outer index */ }
    i = 0;                                         /* rewound: the field is visited again */
  }
  (void)i;

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 852);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = stale ? read_probe(stale) : 0u;    /* the repeated pass's release */
  o->aliased = stale && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "the inner loop reused the outer loop's index, rewinding it over description "
                   "strings already freed and never nulled";
  o->fixed_text = "the fix gives the inner loop its own index, so the outer cursor advances";
  free(fresh);
}
