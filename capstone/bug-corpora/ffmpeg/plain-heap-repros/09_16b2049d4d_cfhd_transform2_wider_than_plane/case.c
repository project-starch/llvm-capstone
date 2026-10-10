#include "corpus.h"

/* cfhd's transform-2 output, fix 16b2049d4d. The plane is ONE direct
 * allocation; here it is the platform's calloc. */

FFH_CASE(9) {
  /* Case 9 -- cfhd transform-2, fix 16b2049d4d. PLAIN HEAP: the inverse
   * transform writes TWICE the lowpass width, and the guard checked only that
   * the lowpass width itself was sane.
   *
   * The guard at the fix's parent ended at:
   *
   *     lowpass_width < 3 || lowpass_height < 3) {
   *
   * and the fix adds the output's own bound:
   *
   *     lowpass_width < 3 || lowpass_height < 3 || lowpass_width * 2 > s->plane[plane].width) {
   *
   * So a stream could declare a lowpass width that passes the lower bound while
   * its DOUBLED output does not fit the plane. The crossing length is a multiple
   * of the declared value, which is why guarding the value alone is not enough.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const long plane_width = 64;     /* the allocated plane row */
  const long lowpass_width = 40;   /* passes `>= 3`; doubled it does not fit */
  CHECK(lowpass_width >= 3, 791);
  CHECK(lowpass_width * 2 > plane_width, 792);   /* the claim, asserted */

  unsigned char *plane = calloc((size_t)plane_width, 1);
  CHECK(plane, 793);

  const long out = lowpass_width * 2;
  long touched = -1;
  if (!fixed) {
    for (long i = 0; i < out && i < plane_width; i++) {
      touched = i;
      plane[i] = (unsigned char)(i & 0xff);
    }

  /* ONE byte past, not the whole overrun. A write loop that really ran the full
   * extent corrupts the next chunk header and glibc aborts in free() -- rc 134,
   * which is an infrastructure failure, not a verdict. The crossing is what the
   * row measures; `extent` records how far the unreduced write would run. This
   * is the same shape case 3 uses. */
    touched = plane_width;
    write_probe_u8(&plane[plane_width], 0x41);   /* the labelled crossing */
  } else {
    touched = -1;   /* the guard rejects the frame before any write */
  }

  o->cap = (unsigned long)plane_width;
  o->touched = touched;
  o->crossed = touched >= plane_width;
  o->extent = out - plane_width;
  o->damage = o->crossed;

  o->defect_text = "the transform-2 guard checked the lowpass width but not its doubled output, so "
                   "the inverse transform wrote lowpass_width * 2 bytes into a narrower plane";
  o->fixed_text = "the fix also rejects lowpass_width * 2 greater than the plane's width, so the "
                  "frame is refused before any write";
  free(plane);
}
