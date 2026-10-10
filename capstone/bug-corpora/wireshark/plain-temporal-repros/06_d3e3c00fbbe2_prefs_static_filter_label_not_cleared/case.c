#include "corpus.h"

/* epan/prefs.c, set_pref's legacy filter-expression path, fix d3e3c00fbbe2.
 * The object is ONE direct g_malloc-family allocation -- no wmem -- which is
 * what puts this row in the plain-temporal corpus rather than ../wmem-repros. */

WST_CASE(6) {
  /* Case 6 -- set_pref's legacy filter label, fix d3e3c00fbbe2. The freed object
   * is the label; what is left holding its address is a FUNCTION-STATIC, so the
   * free and the use are in different calls.
   *
   * The fix clears it:
   *
   *     g_free(filter_label);
   *     filter_label = NULL;
   */
  const unsigned long n = 48;
  unsigned char *label = malloc((size_t)n);
  CHECK(label, 861);
  memset(label, 0x11, (size_t)n);

  /* The function-static, surviving across the two preference entries. */
  volatile unsigned char *filter_label = label;
  free(label);                                    /* the first entry's g_free */
  o->freed = 1;
  if (fixed)
    filter_label = NULL;                          /* the fix's clear */

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 862);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  /* The SECOND PRS_GUI_FILTER_EXPR line, passing the label onward. */
  o->observed = filter_label ? read_probe(filter_label) : 0u;
  o->aliased = filter_label && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "the function-static filter label was freed but left set, so the next legacy "
                   "entry passes storage that now belongs to another object";
  o->fixed_text = "the fix nulls filter_label after freeing it";
  free(fresh);
}
