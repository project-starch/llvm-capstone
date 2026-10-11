#include "corpus.h"

/* ui/cli/tap-wspstat.c, wspstat_init's register-failure path, fix 0fc7f3781351.
 * The object is ONE direct g_malloc-family allocation -- no wmem -- which is
 * what puts this row in the plain-temporal corpus rather than ../wmem-repros. */

WST_CASE(2) {
  /* Case 2 -- wspstat_init's failure path, fix 0fc7f3781351. The freed object is
   * the wspstat_t container, and the later access is a FIELD LOAD out of it:
   * the next two statements both need sp->hash.
   *
   * At the fix's parent:
   *
   *     g_free(sp);
   *     g_hash_table_foreach( sp->hash, (GHFunc) wsp_free_hash_table, NULL ) ;
   *     g_hash_table_destroy( sp->hash );
   *
   * and the fix moves the container's free BELOW the teardown. */
  const unsigned long n = 48;
  const unsigned long hash_off = 8;               /* where the `hash` field sits in sp */
  unsigned char *sp = malloc((size_t)n);
  CHECK(sp, 821);
  memset(sp, 0x11, (size_t)n);

  volatile unsigned char *hash_field = sp + hash_off;
  unsigned char *freed_first = NULL;
  if (!fixed) {
    free(sp);                      /* the container, freed too early */
    freed_first = sp;
    o->freed = 1;
  }

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 822);
  if (freed_first)
    CHECK_REUSE(fresh == freed_first, 823);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = read_probe(hash_field);           /* the teardown's load of sp->hash */
  o->aliased = !fixed && o->observed == o->marker;
  o->damage = o->aliased;
  if (fixed) {
    free(sp);                      /* the fix's order: the container last */
    o->freed = 1;
  }

  o->defect_text = "the container was freed before the hash table it owns was torn down, so the "
                   "teardown loads sp->hash out of storage that now belongs to another object";
  o->fixed_text = "the fix frees the container after the teardown, so every field load is live";
  free(fresh);
}
