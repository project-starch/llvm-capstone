#include "corpus.h"

/* epan/charsets.c, get_t61_string, fix 3aad1ef236e6. The crossed buffer is ONE direct g_malloc-family
 * allocation; no wmem allocator is in the path. */

WSH_CASE(6) {
  /* Case 6 -- get_t61_string's accent branch, fix 3aad1ef236e6. PLAIN HEAP: the
   * branch decodes a TWO-byte sequence and was entered without checking that a
   * second byte exists.
   *
   * At the fix's parent:
   *
   *     for (i = 0, c = ptr; i < length; c++, i++) {
   *         if (!t61_tab[*c]) {
   *             wmem_strbuf_append_unichar(strbuf, UNREPL);
   *         } else if ((*c & 0xf0) == 0xc0) {
   *             gint j = *c & 0x0f;
   *             if ((!c[1] || c[1] == 0x20) && accents[j]) {
   *
   * and the fix requires a following byte:
   *
   *         } else if (i < length - 1 && (*c & 0xf0) == 0xc0) {
   *
   * At the last iteration c[1] is ptr[length], one byte past the input.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long length = 16;                /* a size class */
  unsigned char *ptr = calloc((size_t)length, 1);
  CHECK(ptr, 1221);
  for (unsigned long i = 0; i < length; i++)
    ptr[i] = 0x41;
  ptr[length - 1] = 0xc0;                         /* the accent lead byte, at the LAST position */
  CHECK((ptr[length - 1] & 0xf0) == 0xc0, 1222);  /* the premise, asserted */

  unsigned long touched = 0;
  unsigned acc = 0;
  for (unsigned long i = 0; i < length; i++) {
    const unsigned char c = ptr[i];
    const int accent_branch = fixed ? (i < length - 1 && (c & 0xf0) == 0xc0)
                                    : ((c & 0xf0) == 0xc0);
    if (accent_branch) {
      touched = i + 1;
      if (touched >= length)
        acc += read_probe(ptr + touched);         /* the labelled crossing: c[1] */
      else
        acc += ptr[touched];
    }
  }
  (void)acc;

  o->cap = length;
  o->touched = touched;
  o->crossed = touched >= length;
  o->extent = 1;
  o->damage = o->crossed;
  o->defect_text = "the accent branch decodes a two-byte sequence and was entered on the last byte, "
                   "so c[1] reads ptr[length] -- one past the input";
  o->fixed_text = "the fix requires i < length - 1 before entering the branch";
  free(ptr);
}
