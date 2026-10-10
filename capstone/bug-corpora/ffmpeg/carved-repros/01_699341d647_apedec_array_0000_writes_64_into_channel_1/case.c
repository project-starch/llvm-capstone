#include "corpus.h"

/* apedec's decoded-sample buffer, fix 699341d647 ("prevent out of array writes
 * in decode_array_0000"). ONE allocation carved into the two channels
 * (libavcodec/apedec.c:1485-1491 of the fix's parent):
 *
 *     av_fast_malloc(&s->decoded_buffer, &s->decoded_size,
 *                    2 * FFALIGN(blockstodecode, 8) * sizeof(*s->decoded_buffer));
 *     s->decoded[0] = s->decoded_buffer;
 *     s->decoded[1] = s->decoded_buffer + FFALIGN(blockstodecode, 8);
 *
 * decode_array_0000() writes its first 64 outputs unconditionally -- `for (i =
 * 0; i < 5; i++) out[i] = ...` at :595 and `for (; i < 64; i++) out[i] = ...`
 * at :602 -- so for a frame shorter than 64 samples the write leaves
 * decoded[0] and continues in decoded[1]. For 32 <= blockstodecode < 64 the
 * whole 64-element write lands inside the allocation. The fix bounds both
 * loops by FFMIN(blockstodecode, ...).
 *
 * The MONO path, entropy_decode_mono_0000 (:632-636), which decodes only
 * decoded[0]: the stereo path decodes decoded[1] next with the same loop, whose
 * own overshoot leaves a fresh buffer, and that sibling crossing is not this
 * corpus's defect. The rice decode is reduced to its stores. */
FFC_CASE(1) {
  const int blockstodecode = 40;                  /* a short final frame */
  const size_t stride = FFC_ALIGN(blockstodecode, 8);              /* :1491 */
  const size_t bytes = 2 * stride * sizeof(int32_t);               /* :1486 */
  const int limit = fixed ? (blockstodecode < 64 ? blockstodecode : 64) : 64;

  int32_t *buf = calloc(1, bytes);
  CHECK(buf, 711);
  int32_t *decoded0 = ffc_carve(buf, 0, stride * sizeof(int32_t), "decoded[0]");
  (void)ffc_carve(buf, stride * sizeof(int32_t), stride * sizeof(int32_t), "decoded[1]");
  CHECK(limit <= 64 && (size_t)64 <= 2 * stride, 712); /* the nested window */

  int32_t *out = decoded0;
  int i = 0;
  for (; i < limit && (size_t)i < stride; i++)
    out[i] = 0x100 + i;
  if (i < limit) {
    /* out[stride] is decoded[1][0]: the first element past channel 0. */
    ffc_note(o, buf, bytes, decoded0, stride * sizeof(int32_t), &out[i],
             (size_t)(limit - i) * sizeof(int32_t));
    write_probe_u32((volatile uint32_t *)&out[i], 0x100 + i);
  } else {
    ffc_note(o, buf, bytes, decoded0, stride * sizeof(int32_t), &out[i - 1], sizeof(int32_t));
  }
  /* No consequence on this path: decoded[1] is not read by mono decoding, and
   * on the stereo path its own decode overwrites the spill. */
  o->damage = 0;

  o->defect_text = "decode_array_0000 wrote 64 outputs into a 40-sample channel, so 24 of "
                   "them landed in decoded[1]";
  o->fixed_text = "the fix stops the loop at blockstodecode, the channel's own length";
  free(buf);
}
