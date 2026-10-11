#include "corpus.h"

/* libavfilter/vf_lut3d.c, parse_dat, fix 989444060d5f. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(10) {
  /* Case 10 -- parse_dat's stride, fix 989444060d5f. PLAIN HEAP: size2 was
   * computed from the DEFAULT size before the file's own 3DLUTSIZE line could
   * lower it, while the allocation uses the lowered size.
   *
   * At the fix's parent:
   *
   *     lut3d->lutsize = size = 33;
   *     size2 = size * size;
   *
   *     NEXT_LINE(skip_line(line));
   *     if (!strncmp(line, "3DLUTSIZE ", 10)) {
   *         size = strtol(line + 10, NULL, 0);
   *
   * and the fix moves the stride below the directive:
   *
   *     }
   *     size2 = size * size;
   *
   * With `3DLUTSIZE 2` the array holds 2*2*2 entries and the write index still
   * uses 33*33.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long def_size = 33;              /* the default, before the directive */
  const unsigned long size = 4;                   /* what 3DLUTSIZE declares */
  const unsigned long size2 = fixed ? size * size : def_size * def_size;
  const unsigned long elems = size * size * size; /* 64 entries: 64 B at 1 B/entry */
  CHECK(elems % 16 == 0, 1001);                   /* lands on a size class */
  CHECK(def_size * def_size > elems, 1002);       /* the premise, arm-independent */

  unsigned char *lut = calloc((size_t)elems, 1);
  CHECK(lut, 1003);

  /* The write index the stride produces. One element past is probed; the true
   * index is recorded in `extent`. */
  const long idx = (long)size2;
  const long touched = (idx >= (long)elems) ? (long)elems : idx;
  o->cap = elems;
  o->touched = touched;
  o->crossed = idx >= (long)elems;
  o->extent = idx - (long)elems + 1;
  o->damage = o->crossed;
  if (o->crossed)
    write_probe_u8(lut + touched, 0xA5);          /* the labelled crossing */

  o->defect_text = "size2 was computed from the default LUT size before 3DLUTSIZE could lower it, "
                   "so the write index belongs to a 33-cube while the array is a 4-cube";
  o->fixed_text = "the fix computes size2 after the directive, so the stride matches the allocation";
  free(lut);
}
