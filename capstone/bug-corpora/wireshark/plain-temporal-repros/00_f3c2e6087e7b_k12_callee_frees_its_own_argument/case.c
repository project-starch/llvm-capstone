#include "corpus.h"

/* wiretap/k12.c, k12_open's error paths, fix f3c2e6087e7b.
 * The object is ONE direct g_malloc-family allocation -- no wmem -- which is
 * what puts this row in the plain-temporal corpus rather than ../wmem-repros. */

WST_CASE(0) {
  /* Case 0 -- k12_open's error paths, fix f3c2e6087e7b. The freed object is the
   * k12_t file-state struct; the pointer left holding its address is the
   * caller's own `file_data`, because destroy_k12_file_data frees the struct it
   * is handed and does not clear the caller's variable.
   *
   * At the fix's parent:
   *
   *     destroy_k12_file_data(file_data);
   *     g_free(file_data);
   *
   * and the fix deletes the caller's free. The reduction READS through the stale
   * pointer where upstream FREES through it: a real second free aborts in glibc
   * rather than reporting, and the dangling pointer is the same defect. */
  const unsigned long n = 48;
  unsigned char *file_data = malloc((size_t)n);
  CHECK(file_data, 801);
  memset(file_data, 0x11, (size_t)n);

  volatile unsigned char *stale = file_data;
  free(file_data);                 /* destroy_k12_file_data's own g_free(fd) */
  o->freed = 1;
  if (fixed)
    stale = NULL;                  /* the fix: the caller does not touch it again */

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 802);
  CHECK_REUSE(fresh == file_data, 803);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = stale ? read_probe(stale) : 0u;    /* the caller's second release */
  o->aliased = stale && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "destroy_k12_file_data frees the struct it is handed, so the caller's following "
                   "g_free releases storage that now belongs to another object";
  o->fixed_text = "the fix deletes the caller's free, leaving the helper the single owner";
  free(fresh);
}
