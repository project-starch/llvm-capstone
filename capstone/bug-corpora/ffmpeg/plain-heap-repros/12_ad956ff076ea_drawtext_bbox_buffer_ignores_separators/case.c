#include "corpus.h"

/* libavfilter/vf_drawtext.c, init and the bbox text assembly, fix ad956ff076ea. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(12) {
  /* Case 12 -- drawtext's bbox label buffer, fix ad956ff076ea. PLAIN HEAP: the
   * buffer is sized for N labels of L bytes, and the labels are JOINED with a
   * two-character separator that was never counted.
   *
   * At the fix's parent:
   *
   *     s->text = av_mallocz(AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE *
   *                          (AV_NUM_DETECTION_BBOX_CLASSIFY + 1));
   *
   * and the fix widens each slot by one:
   *
   *     s->text = av_mallocz((AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE + 1) *
   *                          (AV_NUM_DETECTION_BBOX_CLASSIFY + 1));
   *
   * The assembly is strcpy of the detect label then, per classify label,
   * strcat(", ") and strcat(label).
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long label = 8;                  /* AV_DETECTION_BBOX_LABEL_NAME_MAX_SIZE */
  const unsigned long nclass = 1;                 /* AV_NUM_DETECTION_BBOX_CLASSIFY */
  const unsigned long slots = nclass + 1;
  const unsigned long cap = (fixed ? label + 1 : label) * slots;
  CHECK(cap % 16 == 0 || fixed, 1021);            /* the buggy arm lands on a size class: 16 B */

  unsigned char *text = calloc((size_t)cap, 1);
  CHECK(text, 1022);

  /* The assembly: (label-1) chars, then ", " plus (label-1) chars, then the NUL. */
  const unsigned long need = (label - 1) + nclass * (2 + (label - 1)) + 1;
  long touched = 0;
  for (unsigned long i = 0; i < need; i++) {
    touched = (long)i;
    if (i >= cap) {
      write_probe_u8(text + i, (unsigned char)'x');   /* the labelled crossing */
      break;                                          /* one byte past; extent holds the rest */
    }
    text[i] = (unsigned char)'x';
  }

  o->cap = cap;
  o->touched = touched;
  o->crossed = touched >= (long)cap;
  o->extent = (long)need - (long)cap;
  o->damage = o->crossed;

  o->defect_text = "the label buffer was sized for the labels but not for the two-byte separator "
                   "joining them, so the last strcat runs past the allocation";
  o->fixed_text = "the fix widens each label slot by one, which covers the separator and the "
                  "terminator";
  free(text);
}
