#include "corpus.h"

/* epan/prefs.c, prefs_reset, fix 8dc7d164dcdb.
 * The object is ONE direct g_malloc-family allocation -- no wmem -- which is
 * what puts this row in the plain-temporal corpus rather than ../wmem-repros. */

WST_CASE(7) {
  /* Case 7 -- prefs_reset's version field, fix 8dc7d164dcdb. The freed object is
   * the version string; what is left holding its address is the struct field
   * prefs.saved_at_version, and the later access is the SAME function run again.
   *
   * The fix makes the teardown idempotent:
   *
   *     g_free(prefs.saved_at_version);
   *     prefs.saved_at_version = NULL;
   */
  const unsigned long n = 48;
  unsigned char *version = malloc((size_t)n);
  CHECK(version, 871);
  memset(version, 0x11, (size_t)n);

  volatile unsigned char *saved_at_version = version;
  free(version);                                  /* the first prefs_reset */
  o->freed = 1;
  if (fixed)
    saved_at_version = NULL;                      /* the fix's clear */

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 872);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  /* The SECOND prefs_reset, with no read_prefs between. */
  o->observed = saved_at_version ? read_probe(saved_at_version) : 0u;
  o->aliased = saved_at_version && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "prefs_reset freed the version field without clearing it, so a second reset "
                   "reaches storage that now belongs to another object";
  o->fixed_text = "the fix nulls prefs.saved_at_version, making the reset idempotent";
  free(fresh);
}
