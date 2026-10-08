#include "corpus.h"

/* libavcodec/ffv1dec.c, decode_plane's 8-bit remap path, fix 789d7b1b1dff. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(14) {
  /* Case 14 -- decode_plane's 8-bit remap, fix 789d7b1b1dff. PLAIN HEAP: the
   * decoded sample was used as a map index without the mask its two sibling
   * paths apply.
   *
   * At the fix's parent:
   *
   *     sample[1][x] = sc->fltmap[remap_index][sample[1][x]];
   *
   * and the fix:
   *
   *     sample[1][x] = sc->fltmap[remap_index][sample[1][x] & mask];
   *
   * The function asserts the map holds mask+1 entries, so any sample above
   * `mask` -- and a sample is up to 65535 -- reads past it.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long mask = 7;                   /* the map holds mask+1 == 8 entries */
  const unsigned long elems = mask + 1;           /* 8 * 2 B == 16 B: a size class */
  unsigned short *fltmap = calloc((size_t)elems, sizeof *fltmap);
  CHECK(fltmap, 1041);
  for (unsigned long i = 0; i < elems; i++)
    fltmap[i] = (unsigned short)i;

  const unsigned long sample = 300;               /* above mask, as a real sample can be */
  CHECK(sample > mask, 1042);                     /* the premise, asserted */
  const unsigned long idx = fixed ? (sample & mask) : sample;

  const long touched = (idx >= elems) ? (long)elems : (long)idx;
  o->cap = elems;
  o->touched = touched;
  o->crossed = idx >= elems;
  o->extent = (long)idx - (long)elems + 1;
  o->damage = o->crossed;
  if (o->crossed)
    (void)read_probe_u8((const unsigned char *)(fltmap + touched));   /* the labelled crossing */

  o->defect_text = "the 8-bit remap path indexed the map with the raw decoded sample while its two "
                   "sibling paths mask it, so a sample above mask reads past the map";
  o->fixed_text = "the fix masks the index, matching the sibling paths and the map's own assertion";
  free(fltmap);
}
