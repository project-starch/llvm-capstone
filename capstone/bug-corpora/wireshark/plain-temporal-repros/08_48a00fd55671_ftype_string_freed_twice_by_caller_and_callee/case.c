#include "corpus.h"

/* epan/ftypes/ftype-string.c, val_from_unparsed, fix 48a00fd55671.
 * The object is ONE direct g_malloc-family allocation -- no wmem -- which is
 * what puts this row in the plain-temporal corpus rather than ../wmem-repros. */

WST_CASE(8) {
  /* Case 8 -- val_from_unparsed's up-front release, fix 48a00fd55671. The freed
   * object is the string value; what is left holding its address is
   * fv->value.string, because string_fvalue_free does not clear it -- and the
   * second release is in the DELEGATE, val_from_string.
   *
   * At the fix's parent:
   *
   *     string_fvalue_free(fv);           // up front, unconditionally
   *     ...
   *     return val_from_string(fv, s, err_msg);   // frees it again
   *
   * and the fix moves the up-front release into the branch that does NOT
   * delegate. */
  const unsigned long n = 48;
  unsigned char *value = malloc((size_t)n);
  CHECK(value, 881);
  memset(value, 0x11, (size_t)n);

  volatile unsigned char *fv_value_string = value;
  if (!fixed) {
    free(value);                                  /* the up-front string_fvalue_free */
    o->freed = 1;
  } else {
    o->freed = 1;                                 /* the delegate is the single releaser */
  }

  unsigned char *fresh = malloc((size_t)n);
  CHECK(fresh, 882);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = read_probe(fv_value_string);      /* the delegate's own release */
  o->aliased = !fixed && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "the value was released up front and again by the delegate, because "
                   "string_fvalue_free does not clear the field it frees";
  o->fixed_text = "the fix releases only in the branch that does not delegate, leaving one owner";
  free(fresh);
  if (fixed)
    free(value);
}
