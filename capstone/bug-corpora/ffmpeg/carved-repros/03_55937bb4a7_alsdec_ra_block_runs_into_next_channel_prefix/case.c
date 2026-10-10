#include "corpus.h"

/* alsdec's raw sample buffer, fix 55937bb4a7 ("fix address sanitization error
 * in decoder"). ONE allocation carved into channels, each a max_order
 * carry-over prefix followed by frame_length samples (libavcodec/alsdec.c of
 * the fix's parent):
 *
 *     channel_size     = sconf->frame_length + sconf->max_order;          // :2056
 *     ctx->raw_buffer  = av_mallocz_array(avctx->channels * channel_size,
 *                                         sizeof(*ctx->raw_buffer));      // :2059
 *     ctx->raw_samples[0] = ctx->raw_buffer + sconf->max_order;           // :2096
 *     ctx->raw_samples[c] = ctx->raw_samples[c - 1] + channel_size;       // :2098
 *
 * The random-access block reconstructs its first opt_order samples in a loop
 * bounded by opt_order alone (:922-931):
 *
 *     for (smp = 0; smp < opt_order; smp++) {
 *         ...
 *         *raw_samples++ -= y >> 20;
 *
 * so a block shorter than opt_order is read-modified-written past its end. With
 * a non-adaptive order (opt_order = max_order, :666-667) larger than the frame
 * length, the overshoot leaves channel 0 and lands in channel 1's carry-over
 * prefix -- the samples channel 1's next prediction starts from. It is shorter
 * than that prefix, so it never reaches channel 1's samples, and for every
 * channel but the last it stays inside the allocation. The fix bounds the loop
 * by FFMIN(opt_order, block_length).
 *
 * The prediction is reduced to the decrement it applies (y >> 20 taken as 1),
 * so the read-modify-write is visible in channel 1's prefix. */
FFC_CASE(3) {
  const size_t frame_length = 16, max_order = 24, channels = 2;
  const size_t channel_size = frame_length + max_order;          /* :2056 */
  const size_t bytes = channels * channel_size * sizeof(int32_t); /* :2059 */
  const size_t block_length = frame_length; /* one RA block per frame */
  const size_t opt_order = max_order;       /* non-adaptive, :667 */
  const size_t limit = fixed ? (opt_order < block_length ? opt_order : block_length)
                             : opt_order;

  int32_t *raw_buffer = calloc(channels * channel_size, sizeof(int32_t));
  CHECK(raw_buffer, 731);
  int32_t *chan[2], *raw_samples[2];
  for (size_t c = 0; c < channels; c++) {
    chan[c] = ffc_carve(raw_buffer, c * channel_size * sizeof(int32_t),
                        channel_size * sizeof(int32_t), c ? "channel[1]" : "channel[0]");
    raw_samples[c] = chan[c] + max_order; /* :2096-2098 */
  }
  CHECK(opt_order - block_length < max_order, 732); /* stays in channel 1's prefix */
  for (size_t k = 0; k < max_order; k++)
    chan[1][k] = 1000 + (int32_t)k; /* channel 1's carry-over from its previous frame */
  for (size_t k = 0; k < frame_length; k++)
    raw_samples[0][k] = 10 * (int32_t)k;

  int32_t *p = raw_samples[0];
  size_t smp = 0;
  for (; smp < limit && smp < block_length; smp++)
    *p++ -= 1;
  if (smp < limit) {
    /* raw_samples[0][block_length] is the first word of channel 1's prefix. */
    ffc_note(o, raw_buffer, bytes, chan[0], channel_size * sizeof(int32_t), p,
             (limit - smp) * sizeof(int32_t));
    uint32_t v = read_probe_u32((const volatile uint32_t *)p);
    *p = (int32_t)v - 1;
  } else {
    ffc_note(o, raw_buffer, bytes, chan[0], channel_size * sizeof(int32_t), p - 1,
             sizeof(int32_t));
  }
  o->damage = chan[1][0] != 1000; /* channel 1's next prediction starts from it */

  o->defect_text = "the RA loop reconstructed 24 samples of a 16-sample block, so it "
                   "decremented the first of channel 1's carry-over samples";
  o->fixed_text = "the fix stops the loop at block_length";
  free(raw_buffer);
}
