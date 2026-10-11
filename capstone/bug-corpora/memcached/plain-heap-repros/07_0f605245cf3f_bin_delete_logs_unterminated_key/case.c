#include "corpus.h"

/* process_bin_delete's verbose branch, fix 0f605245cf3f. The key region lies
 * inside the connection read buffer, one direct allocation; neither slabs.c nor
 * cache.c is involved. */

MCH_CASE(7) {
  /* Case 7 -- the binary delete's verbose log, fix 0f605245cf3f. PLAIN HEAP:
   * the key is a COUNTED string and was printed as a C string.
   *
   * At the fix's parent:
   *
   *     if (settings.verbose > 1) {
   *         fprintf(stderr, "Deleting %s\n", key);
   *     }
   *
   * and the fix prints exactly nkey bytes:
   *
   *     int ii;
   *     fprintf(stderr, "Deleting ");
   *     for (ii = 0; ii < nkey; ++ii) {
   *         fprintf(stderr, "%c", key[ii]);
   *     }
   *
   * `key` is `binary_get_key(c)` == `c->rcurr - keylen`, a region inside the
   * read buffer delimited by nkey and by nothing else. The "%s" conversion walks
   * until it finds a zero byte, which may lie past the allocation.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long nkey = 16;            /* a size class: the key fills the allocation exactly */
  unsigned char *key = calloc((size_t)nkey, 1);
  CHECK(key, 951);
  for (unsigned long i = 0; i < nkey; i++)
    key[i] = (unsigned char)('a' + (i % 26));   /* NO terminator anywhere in it */
  CHECK(key[nkey - 1] != 0, 952);               /* the claim, asserted */

  /* The scan the "%s" conversion performs, bounded in the fixed arm by nkey. */
  unsigned long touched = 0;
  unsigned acc = 0;
  for (unsigned long i = 0; ; i++) {
    if (fixed && i >= nkey)
      break;                                /* the fix's counted loop */
    touched = i;
    if (i >= nkey) {
      acc += read_probe(key + i);           /* the labelled crossing */
      break;                                /* the reduction's own stop */
    }
    if (key[i] == 0)
      break;
    acc += key[i];
  }
  (void)acc;

  o->cap = nkey;
  o->touched = touched;
  o->crossed = touched >= nkey;
  o->damage = o->crossed;

  o->defect_text = "the counted key was printed with \"%s\", so the conversion scanned past the "
                   "nkey bytes that are the key and read beyond the allocation";
  o->fixed_text = "the fix prints exactly nkey bytes with a counted loop, so the scan stays inside";
  free(key);
}
