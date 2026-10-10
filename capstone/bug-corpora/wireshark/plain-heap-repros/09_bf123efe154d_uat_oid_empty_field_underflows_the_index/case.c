#include "corpus.h"

/* epan/uat.c, uat_fld_chk_oid, fix bf123efe154d. The crossed buffer is ONE direct g_malloc-family
 * allocation; no wmem allocator is in the path. */

WSH_CASE(9) {
  /* Case 9 -- uat_fld_chk_oid's trailing check, fix bf123efe154d. PLAIN HEAP:
   * with an empty field the validation loop does not execute and the code falls
   * into strptr[len-1].
   *
   * The fix rejects the empty field first:
   *
   *     if (len == 0) {
   *       *err = g_strdup("Empty OID");
   *       return FALSE;
   *     }
   *
   * `len` is a guint, so `len - 1` wraps to 4294967295 BEFORE the pointer
   * arithmetic -- a read about 4 GiB past the buffer, not one byte before it.
   * The probe touches the first byte OUTSIDE the allocation and the true offset
   * is recorded in `extent`, because dereferencing 4 GiB out is a wild access
   * rather than a reportable crossing.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long cap = 16;                   /* a size class */
  unsigned char *strptr = calloc((size_t)cap, 1);
  CHECK(strptr, 1251);

  const unsigned long len = 0;                    /* the empty field */
  CHECK(len == 0, 1252);                          /* the premise, asserted */

  unsigned long touched = 0;
  int crossed = 0;
  if (!fixed) {
    /* strptr[len-1] with len a 32-bit unsigned: the index wraps. */
    const unsigned long idx = (unsigned long)(unsigned int)(len - 1u);
    CHECK(idx > cap, 1253);                       /* it really did wrap */
    touched = cap;                                /* the probe stays at the first byte outside */
    crossed = 1;
    (void)read_probe(strptr + cap);               /* the labelled crossing */
    o->extent = (long)idx;                        /* the true offset: 4294967295 */
  } else {
    o->extent = (long)4294967295u;
  }

  o->cap = cap;
  o->touched = touched;
  o->crossed = crossed;
  o->damage = crossed;
  o->defect_text = "an empty OID field skipped the loop and fell into strptr[len-1], whose guint "
                   "index wraps to 4294967295 before the pointer arithmetic";
  o->fixed_text = "the fix rejects a zero-length field before the trailing-character check";
  free(strptr);
}
