#include "corpus.h"

/* apedec's long_filter_high_3800, fix cd7524fdd1 ("Check length in
 * long_filter_high_3800()"). The buffer is the same carved decoded_buffer as
 * case 1 (libavcodec/apedec.c:1485-1491 of the parent): decoded[0] is
 * FFALIGN(count, 8) samples and decoded[1] follows it.
 *
 * The filter primes its delay line from the first `order` samples without
 * looking at the length (:889-897):
 *
 *     for (i = 0; i < order; i++)
 *         delay[i] = buffer[i];
 *
 * and the mono 3800 predictor calls it with order 128 at the extra-high level
 * (predictor_decode_mono_3800, :997-1006). For a frame of fewer than 128
 * samples the read leaves decoded[0] and continues in decoded[1]; with order <=
 * 2 * FFALIGN(count, 8) all of it stays inside the allocation. The fix returns
 * when order >= length.
 *
 * READ ONLY, and inert: when order >= length the filter loop after it does not
 * run, so delay[] is never used. The crossing is real; the corruption is not.
 * The MONO path again, for case 1's reason: the stereo predictor runs the same
 * filter over decoded[1], whose read leaves a fresh buffer. */
FFC_CASE(2) {
  const int count = 72, order = 128;              /* extra-high, version < 3830 */
  const size_t stride = FFC_ALIGN(count, 8);
  const size_t bytes = 2 * stride * sizeof(int32_t);

  int32_t *buf = calloc(1, bytes);
  CHECK(buf, 721);
  int32_t *decoded0 = ffc_carve(buf, 0, stride * sizeof(int32_t), "decoded[0]");
  (void)ffc_carve(buf, stride * sizeof(int32_t), stride * sizeof(int32_t), "decoded[1]");
  CHECK((size_t)order <= 2 * stride, 722); /* the nested window */
  for (int i = 0; i < count; i++)
    decoded0[i] = i;

  if (fixed && order >= count) {
    /* The fix: `if (order >= length) return;` before the delay line. */
  } else {
    int32_t delay[256];
    int i = 0;
    for (; i < order && (size_t)i < stride; i++)
      delay[i] = decoded0[i];
    ffc_note(o, buf, bytes, decoded0, stride * sizeof(int32_t), &decoded0[i],
             (size_t)(order - i) * sizeof(int32_t));
    delay[i] = (int32_t)read_probe_u32((const volatile uint32_t *)&decoded0[i]);
    (void)delay;
  }
  o->damage = 0; /* inert: the loop that would use delay[] does not run */

  o->defect_text = "the delay line read 128 samples from a 72-sample channel, 56 of them "
                   "from decoded[1]";
  o->fixed_text = "the fix returns before the delay line when order >= length";
  free(buf);
}
