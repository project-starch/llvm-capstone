#include "corpus.h"

/* libavcodec/exif.c, exif_clone_entry's AV_TIFF_STRING case, fix 76645e096fab. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(11) {
  /* Case 11 -- exif_clone_entry's string case, fix 76645e096fab. PLAIN HEAP:
   * the source string is allocated count+1 bytes and NUL-terminated, and the
   * clone copied only `count`.
   *
   * At the fix's parent:
   *
   *     case AV_TIFF_STRING:
   *         EXIF_COPY(dst->value.str, src->value.str);
   *
   * where EXIF_COPY computes `sz = src->count * sizeof(*fname)` -- for a char*
   * that is `count`. The fix copies one more:
   *
   *         dst->value.str = av_memdup(src->value.str, src->count+1);
   *
   * The clone is then an unterminated count-byte allocation, and the first
   * strlen or "%s" on it runs off the end.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long count = 16;                 /* a size class for the clone */
  unsigned char *src = calloc((size_t)count + 1, 1);   /* av_mallocz(count + 1) */
  CHECK(src, 1011);
  for (unsigned long i = 0; i < count; i++)
    src[i] = (unsigned char)('a' + (i % 26));
  src[count] = 0;                                 /* the terminator the source carries */

  const unsigned long n = fixed ? count + 1 : count;
  unsigned char *clone = calloc((size_t)n, 1);    /* av_memdup's allocation */
  CHECK(clone, 1012);
  memcpy(clone, src, (size_t)n);

  /* The consumer's scan: strlen over the clone. */
  long touched = 0;
  unsigned acc = 0;
  for (unsigned long i = 0; ; i++) {
    touched = (long)i;
    if (i >= n) {
      acc += read_probe_u8(clone + i);            /* the labelled crossing */
      break;
    }
    if (clone[i] == 0)
      break;
    acc += clone[i];
  }
  (void)acc;

  o->cap = n;
  o->touched = touched;
  o->crossed = touched >= (long)n;
  o->extent = 1;
  o->damage = o->crossed;

  o->defect_text = "the clone copied only the character count, so it carries no terminator and the "
                   "first scan over it runs past the allocation";
  o->fixed_text = "the fix copies count+1, inheriting the terminator the source was allocated for";
  free(clone);
  free(src);
}
