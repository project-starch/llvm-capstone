#include "corpus.h"

/* vf_scale's colour-space list compaction, fix bcbf3a5630. The list is ONE
 * direct allocation: ff_all_color_spaces() builds it through MAKE_FORMAT_LIST,
 * whose body is `formats->field = av_malloc_array(count, sizeof(*formats->field))`
 * at libavfilter/formats.c:429. Here it is the platform's calloc. */

FFH_CASE(1) {
  /* Case 1 -- vf_scale query_formats, fix bcbf3a5630. PLAIN HEAP: a four-byte
   * read one element past a direct av_malloc_array, feeding a write inside it.
   *
   * The compaction loop at the pin (vf_scale.c:483-487, and again at :503-507):
   *
   *     for (int i = 0; i < formats->nb_formats; i++) {
   *         if (!sws_test_colorspace(formats->formats[i], 0)) {
   *             for (int j = i--; j < formats->nb_formats; j++)
   *                 formats->formats[j] = formats->formats[j + 1];
   *             formats->nb_formats--;
   *         }
   *     }
   *
   * The shift loop's guard is `j < nb_formats` while its body reads index
   * `j + 1`, so the last iteration has `j == nb_formats - 1` and reads
   * `formats[nb_formats]` -- one past the array. The fix changes the guard to
   * `j + 1 < formats->nb_formats`.
   *
   * Upstream's own note is worth preserving because it is exactly the kind of
   * thing this corpus exists to measure: "Fortunately, the excess element was
   * never actually used, but it still triggers ASAN (and could in theory trigger
   * a segfault)." The value is discarded -- the write that consumes it lands on
   * the element being removed, which the next `nb_formats--` puts out of range.
   * So the DAMAGE is nil and the CROSSING is real. A bounds-checking machine
   * faults on the read regardless of what happens to the value, which is why
   * "never actually used" is not a defence and why the fix was taken.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const long count = 6;
  int *formats = calloc((size_t)count, sizeof *formats);
  CHECK(formats, 711);
  for (long i = 0; i < count; i++)
    formats[i] = (int)(100 + i);

  long nb_formats = count;
  /* The element that fails the colour-space test. Chosen as the FIRST one, so
   * the shift loop runs its full length and reaches the last index. */
  const long reject = 0;

  long touched = -1;
  for (long i = 0; i < nb_formats; i++) {
    if (i != reject)
      continue;
    long start = i--;
    for (long j = start; fixed ? (j + 1 < nb_formats) : (j < nb_formats); j++) {
      if (j + 1 > touched)
        touched = j + 1;
      /* The out-of-bounds read is the j+1 index; probe it when it is the one
       * past the end so a report can be required at this line. */
      if (j + 1 == count)
        formats[j] = (int)read_probe_u8((const volatile unsigned char *)&formats[j + 1]);
      else
        formats[j] = formats[j + 1];
    }
    nb_formats--;
  }

  o->cap = (unsigned long)count;
  o->touched = touched;
  o->crossed = touched >= count;
  o->extent = 1;
  /* Upstream's point: the crossed value is discarded, so there is no damage. */
  o->damage = 0;

  o->defect_text = "the shift loop's guard was j < nb_formats while its body read "
                   "formats[j + 1], so it read one element past av_malloc_array -- "
                   "the value is discarded, the crossing is not";
  o->fixed_text = "the fix's j + 1 < nb_formats guard keeps every read inside the array";
  free(formats);
}
