#include "corpus.h"

/* The display-filter VM's binary operator walks two value arrays with two
 * independent indices and then, on the error path only, indexes the SECOND
 * array with the FIRST array's index.
 *
 * A GPtrArray's `pdata` is one direct g_malloc'd pointer array -- GLib grows it
 * with g_realloc and nothing sub-allocates it -- so the overread crosses the
 * malloc bound itself. That is this corpus's boundary, and it is why the row is
 * here rather than in ../wmem-repros. GLib is not linked (see corpus.h);
 * g_malloc is malloc plus abort-on-failure, which is all this uses of it. */

WSH_CASE(1) {
  /* Case 1 -- dfvm binary operator error path, fix 373504f7c9 ("dfvm: Fix an
   * error message to avoid an out-of-bounds read"). PLAIN HEAP, and LIVE at the
   * v4.6.8 pin.
   *
   * mk_binary_internal at the pin (epan/dfilter/dfvm.c:1382-1389):
   *
   *     for (size_t i = 0; i < fv1->len; i++) {
   *         for (size_t j = 0; j < fv2->len; j++) {
   *             result = func(fv1->pdata[i], fv2->pdata[j], &err_msg);
   *             if (result == NULL) {
   *                 debug_op_error(fv1->pdata[i], fv2->pdata[i], "&", err_msg);
   *
   * The call that DOES the work indexes correctly -- `fv2->pdata[j]`. The error
   * message, reached only when the operation fails, indexes `fv2` with **i**.
   * The two arrays have independent lengths, so whenever fv1 is longer than fv2
   * and the operation fails at an i beyond fv2's end, the read is past
   * `fv2->pdata`.
   *
   * The fix is one character: `i` becomes `j`.
   *
   * WHY THIS ROW IS WORTH HAVING beyond the crossing: the defect is on the
   * ERROR PATH ONLY. The correct index is three lines above it, in the same
   * statement block, which is why it survived review and why it took a
   * dfilter-syntax test to find. A reduction that exercised only the success
   * path would report a clean result and learn nothing -- the arms here
   * deliberately force the failure.
   *
   * THE CROSSING LEAVES THE ALLOCATION: a per-object bound is NOT in bounds for
   * it, so ASan sees it and so should a capability machine. */
  const unsigned long fv1_len = 4;
  const unsigned long fv2_len = 2; /* shorter: this asymmetry is the defect's premise */

  void **fv1 = malloc(fv1_len * sizeof *fv1);
  CHECK(fv1, 901);
  void **fv2 = malloc(fv2_len * sizeof *fv2);
  CHECK(fv2, 902);
  for (unsigned long k = 0; k < fv1_len; k++)
    fv1[k] = (void *)(unsigned long)(0x100 + k);
  for (unsigned long k = 0; k < fv2_len; k++)
    fv2[k] = (void *)(unsigned long)(0x200 + k);

  /* The failing iteration: i has run past fv2's end. The loop structure means
   * the inner j is always in range; only the error message's index is not. */
  const unsigned long i = fv1_len - 1; /* 3, past fv2_len - 1 */
  const unsigned long j = fv2_len - 1; /* 1, in range */

  /* Both halves of the claim, asserted rather than assumed: the error path's
   * index is past fv2, and the working call's index is not. A reduction that
   * lost the asymmetry would fail here instead of reporting a verdict. */
  CHECK(i >= fv2_len, 903);
  CHECK(j < fv2_len, 904);

  o->cap = fv2_len;
  o->touched = i;
  o->crossed = i >= fv2_len;
  /* Unreduced, the read runs as far past as fv1 is longer than fv2. */
  o->extent = (long)(fv1_len - fv2_len);

  /* The error message's read. Under the fix the index is j and stays inside. */
  unsigned long index = fixed ? j : i;
  o->touched = index;
  o->crossed = index >= fv2_len;
  (void)read_probe((const volatile unsigned char *)&fv2[index]);
  /* The value is only formatted into a debug string, so there is no damage
   * beyond the read itself -- which is the point: a consequence-based test
   * cannot see this defect, a bounds check can. */
  o->damage = 0;

  o->defect_text = "the error path indexed fv2 with fv1's index i, so the read ran past "
                   "fv2->pdata -- one direct g_malloc -- while the working call three "
                   "lines above used j correctly";
  o->fixed_text = "the fix's one-character change from i to j keeps the error path's read "
                  "inside fv2";
  free(fv2);
  free(fv1);
}
