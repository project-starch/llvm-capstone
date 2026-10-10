#include "corpus.h"

/* do_suffix_add_to_freelist, fix 391f2e4762bf. The freelist is ONE direct
 * malloc/realloc of char*; neither slabs.c nor cache.c is involved. */

MCH_CASE(2) {
  /* Case 2 -- the suffix freelist's grow path, fix 391f2e4762bf. PLAIN HEAP:
   * the realloc was sized in BYTES where the capacity it sets is in ELEMENTS.
   *
   * At the fix's parent:
   *
   *     char **new_freesuffix = realloc(freesuffix, freesuffixtotal * 2);
   *     if (new_freesuffix) {
   *         freesuffixtotal *= 2;
   *         freesuffix = new_freesuffix;
   *         freesuffix[freesuffixcurr++] = s;
   *
   * and the fix multiplies by the element size:
   *
   *     char **new_freesuffix = realloc(freesuffix,
   *         sizeof(char *) * freesuffixtotal * 2);
   *
   * So the array is asked to hold `total * 2` BYTES -- room for total/4
   * pointers on a 64-bit target -- while `freesuffixtotal` is set to `total * 2`
   * POINTERS, and the store at index `total` follows immediately.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long total = 8;                      /* elements, as upstream counts them */
  const unsigned long grown = fixed ? sizeof(char *) * total * 2   /* 128 B: 16 pointers */
                                    : total * 2;                   /* 16 B: 2 pointers */
  CHECK(grown % 16 == 0, 901);  /* both arms land on a size class: no slack to absorb a crossing */

  char **fs = malloc(sizeof(char *) * total);
  CHECK(fs, 902);
  for (unsigned long i = 0; i < total; i++)
    fs[i] = NULL;

  char **re = realloc(fs, (size_t)grown);
  CHECK(re, 903);

  /* Upstream's next statement stores at index `total`, i.e. byte offset
   * sizeof(char*) * total == 64 (128 on purecap) -- 48 (112) bytes past the buggy arm's 16-byte
   * allocation. The probe touches the FIRST byte past instead: writing all 48
   * smashes the next chunk header and glibc aborts in free(), which is not a
   * verdict. The true distance is in defect_text. */
  const unsigned long store_off = sizeof(char *) * total;   /* 64; 128 on purecap */
  const unsigned long touched = (store_off >= grown) ? grown : store_off;
  o->cap = grown;
  o->touched = touched;
  o->crossed = store_off >= grown;
  o->damage = o->crossed;
  if (o->crossed)
    write_probe((volatile unsigned char *)re + touched, 0xA5);  /* the labelled crossing */

  o->defect_text = "the freelist was reallocated to freesuffixtotal*2 BYTES while the capacity was "
                   "set to that many POINTERS, so the next store lands 48 bytes past a 16-byte "
                   "allocation";
  o->fixed_text = "the fix multiplies by sizeof(char *), so the array really holds the capacity it "
                  "claims";
  free(re);
}
