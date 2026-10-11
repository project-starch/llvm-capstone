#include "corpus.h"

/* pcapng's NRB custom-string option, fix 381681583b. The string is ONE direct
 * allocation; wiretap does not use wmem, which is what puts this row in the
 * plain-heap corpus rather than the wmem one. */

WSH_CASE(2) {
  /* Case 2 -- pcapng put_nrb_option, fix 381681583b. PLAIN HEAP: the copy was
   * sized by the OPTION's value size, which by the pcapng layout is the PEN
   * plus the string:
   *
   *     stringlen = strlen(optval->custom_stringval.string);
   *     size = sizeof(uint32_t) + stringlen;
   *
   * The PEN is written separately, just before:
   *
   *     memcpy(*opt_ptrp, &optval->custom_stringval.pen, sizeof(uint32_t));
   *     *opt_ptrp += sizeof(uint32_t);
   *     memcpy(*opt_ptrp, optval->custom_stringval.string, size);   <-- the defect
   *
   * so `size` bytes are taken out of a buffer that holds only the string. The
   * fix copies `stringlen`. The source allocation is strlen+1 bytes -- the
   * string and its NUL -- and the copy asks for strlen+4, so 3 bytes lie past
   * the end of the allocation.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  /* stringlen is chosen so the allocation lands EXACTLY on a 16-byte jemalloc size
   * class: with slack, the 3-byte crossing would be absorbed and CheriBSD would read
   * clean for a reason that has nothing to do with the defect. */
  const unsigned long stringlen = 15;
  const unsigned long alloc = stringlen + 1;      /* 16: the string and its NUL */
  const unsigned long size = sizeof(uint32_t) + stringlen;   /* upstream's option size: 19 */
  CHECK(size > alloc, 811);                       /* the claim, asserted */

  unsigned char *str = calloc((size_t)alloc, 1);
  CHECK(str, 812);
  for (unsigned long i = 0; i < stringlen; i++)
    str[i] = (unsigned char)('a' + (i % 26));
  str[stringlen] = 0;

  unsigned char *dst = calloc((size_t)size, 1);
  CHECK(dst, 813);

  const unsigned long n = fixed ? stringlen : size;
  unsigned long touched = 0;
  for (unsigned long i = 0; i < n; i++) {
    touched = i;
    if (i >= alloc)
      dst[i] = (unsigned char)read_probe(&str[i]);   /* the labelled crossing */
    else
      dst[i] = str[i];
  }

  o->cap = alloc;
  o->touched = touched;
  o->crossed = touched >= alloc;
  o->extent = (long)(size - alloc);                /* 19 - 16 = 3 */
  o->damage = o->crossed;

  o->defect_text = "the custom-string option was copied by the option's value size, which counts the "
                   "4-byte PEN the string buffer does not hold, so the copy read past the string";
  o->fixed_text = "the fix copies stringlen, the string's own length, so the read stays inside it";
  free(dst);
  free(str);
}
