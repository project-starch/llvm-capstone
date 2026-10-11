#include "corpus.h"

/* ui/commandline.c, cl_find_custom, fix 0cae98570ebc. The crossed buffer is ONE direct g_malloc-family
 * allocation; no wmem allocator is in the path. */

WSH_CASE(8) {
  /* Case 8 -- cl_find_custom's comparison, fix 0cae98570ebc. PLAIN HEAP: the
   * length comes from the SEARCH prefix and is applied to both operands.
   *
   * At the fix's parent:
   *
   *     static int cl_find_custom(const void *elem_data, const void *search_data) {
   *         return memcmp(elem_data, search_data, strlen((char *)search_data));
   *     }
   *
   * and the fix compares as strings:
   *
   *         return strncmp(opt_and_val, prefix, strlen(prefix));
   *
   * memcmp is SPECIFIED to read n bytes from both operands, so a stored option
   * shorter than the prefix is read past its end -- the reduction performs that
   * read explicitly rather than relying on a particular memcmp implementation.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long opt_len = 16;               /* the stored option, a size class */
  unsigned char *opt = calloc((size_t)opt_len, 1);
  CHECK(opt, 1241);
  for (unsigned long i = 0; i < opt_len - 1; i++)
    opt[i] = (unsigned char)('a' + (i % 26));
  opt[opt_len - 1] = 0;                           /* a proper C string */

  const unsigned long prefix_len = 24;            /* LONGER than the stored option */
  CHECK(prefix_len > opt_len, 1242);              /* the premise, asserted */

  unsigned long touched = 0;
  unsigned acc = 0;
  const unsigned long n = fixed ? opt_len : prefix_len;   /* strncmp stops at the NUL */
  for (unsigned long i = 0; i < n; i++) {
    touched = i;
    if (fixed && opt[i] == 0)
      break;                                      /* strncmp's own stop */
    if (i >= opt_len) {
      acc += read_probe(opt + i);                 /* the labelled crossing */
      break;
    }
    acc += opt[i];
  }
  (void)acc;

  o->cap = opt_len;
  o->touched = touched;
  o->crossed = touched >= opt_len;
  o->extent = (long)(prefix_len - opt_len);
  o->damage = o->crossed;
  o->defect_text = "memcmp was given the search prefix's length and is specified to read that many "
                   "bytes from BOTH operands, so a shorter stored option is read past its end";
  o->fixed_text = "the fix uses strncmp, which stops at the stored option's terminator";
  free(opt);
}
