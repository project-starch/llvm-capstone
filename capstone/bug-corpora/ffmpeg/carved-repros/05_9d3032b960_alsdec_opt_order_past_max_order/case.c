#include "corpus.h"

/* alsdec's quantised PARCOR coefficients, fix 9d3032b960 ("check opt_order").
 * ONE allocation carved per channel in MCC mode (libavcodec/alsdec.c of the
 * fix's parent):
 *
 *     num_buffers = sconf->mc_coding ? avctx->channels : 1;                 // :1628
 *     ctx->quant_cof_buffer = av_malloc(sizeof(*ctx->quant_cof_buffer) *
 *                                       num_buffers * sconf->max_order);   // :1632-1633
 *     ctx->quant_cof[c] = ctx->quant_cof_buffer + c * sconf->max_order;    // :1648
 *
 * The adaptive order is read in a field sized from max_order + 1, not bounded
 * by it (:663-665):
 *
 *     int opt_order_length = av_ceil_log2(av_clip((bd->block_length >> 3) - 1,
 *                                         2, sconf->max_order + 1));
 *     *bd->opt_order = get_bits(gb, opt_order_length);
 *
 * With max_order 20 the field is five bits, so opt_order can be 31, and the
 * coefficient loops (:693-711) write quant_cof[k] for every k < opt_order: 11
 * words past channel 0's slice, into channel 1's. In MCC mode with two
 * channels the whole overshoot stays inside the allocation (without MCC there
 * is one slice and it escapes). The fix returns an error when opt_order >
 * max_order.
 *
 * Channel 1's coefficients are read after channel 0's, so they overwrite the
 * spill; channel 0's own prediction then uses channel 1's coefficients 20..30
 * as its own. The rice decode is reduced to its stores. */
FFC_CASE(5) {
  const unsigned channels = 2, max_order = 20, block_length = 256;
  const size_t bytes = sizeof(int32_t) * channels * max_order;
  unsigned clip = (block_length >> 3) - 1;
  clip = clip < 2 ? 2 : clip > max_order + 1 ? max_order + 1 : clip;
  unsigned opt_order_length = 0;
  while ((1u << opt_order_length) < clip)
    opt_order_length++;                                 /* av_ceil_log2 */
  const unsigned opt_order = (1u << opt_order_length) - 1; /* the largest field value */
  CHECK(opt_order == 31 && opt_order <= 2 * max_order, 751); /* the nested window */

  int32_t *quant_cof_buffer = calloc(channels * max_order, sizeof(int32_t));
  CHECK(quant_cof_buffer, 752);
  int32_t *quant_cof[2];
  for (unsigned c = 0; c < channels; c++)
    quant_cof[c] = ffc_carve(quant_cof_buffer, c * max_order * sizeof(int32_t),
                             max_order * sizeof(int32_t), c ? "quant_cof[1]" : "quant_cof[0]");

  if (fixed && opt_order > max_order) {
    /* The fix: `if (*bd->opt_order > sconf->max_order) return -1;` -- the
     * block is refused before any coefficient is read. */
  } else {
    unsigned k = 0;
    for (; k < opt_order && k < max_order; k++)
      quant_cof[0][k] = 0x100 + (int32_t)k;
    ffc_note(o, quant_cof_buffer, bytes, quant_cof[0], max_order * sizeof(int32_t),
             &quant_cof[0][k], (opt_order - k) * sizeof(int32_t));
    write_probe_u32((volatile uint32_t *)&quant_cof[0][k], 0x100 + k);
    for (k++; k < opt_order; k++)
      quant_cof[0][k] = 0x100 + (int32_t)k;
    /* Channel 1's block: its own 20 coefficients. */
    for (k = 0; k < max_order; k++)
      quant_cof[1][k] = 0x200 + (int32_t)k;
    o->damage = quant_cof[0][max_order] != 0x100 + (int32_t)max_order;
  }

  o->defect_text = "opt_order 31 wrote 11 coefficients past channel 0's 20-word slice, into "
                   "channel 1's";
  o->fixed_text = "the fix refuses a block whose opt_order exceeds max_order";
  free(quant_cof_buffer);
}
