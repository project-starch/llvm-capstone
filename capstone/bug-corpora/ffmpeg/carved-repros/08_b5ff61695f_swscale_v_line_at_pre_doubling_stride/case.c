#include "corpus.h"

/* swscale's chroma line buffers, fix b5ff61695f ("fix uv overwrite in 32bt").
 * Each chroma line is ONE allocation holding U and then V (libswscale/utils.c
 * of the fix's parent):
 *
 *     int dst_stride = FFALIGN(dstW * sizeof(int16_t)+66, 16), dst_stride_px = dst_stride >> 1;  // :789
 *     if (c->scalingBpp == 16)
 *         dst_stride <<= 1;                                                   // :889-890
 *     FF_ALLOC_OR_GOTO(c, c->chrUPixBuf[i+c->vChrBufSize], dst_stride*2+1, fail);              // :1053
 *     c->chrVPixBuf[i] = ... = c->chrUPixBuf[i] + dst_stride_px;              // :1055
 *
 * dst_stride_px is taken BEFORE the doubling, so with 16-bit scaling V starts
 * half a doubled stride too early. The horizontal scaler writes one int32 per
 * chroma pixel into U, and when chroma is not horizontally subsampled (4:4:4
 * output) that is 4 * dstW bytes, more than the 2 * dstW + 66 rounded to 16
 * that V's offset left. U's tail lands in V, V's own write then overwrites it,
 * and the vertical scaler reads V's samples as U's. The fix places V at
 * `dst_stride >> 1` elements, the doubled stride's half.
 *
 * Both writes stay inside the line: V ends at S0 + 4 * dstW < 4 * S0 + 1. */
FFC_CASE(8) {
  const size_t dstW = 64, chrDstW = dstW;              /* 4:4:4 output */
  size_t dst_stride = FFC_ALIGN(dstW * sizeof(int16_t) + 66, 16); /* :789, 208 */
  const size_t dst_stride_px = dst_stride >> 1;        /* taken before the doubling */
  dst_stride <<= 1;                                    /* scalingBpp == 16, :889-890 */
  const size_t bytes = dst_stride * 2 + 1;             /* :1053 */
  const size_t v_off = (fixed ? dst_stride >> 1 : dst_stride_px) * sizeof(int16_t); /* :1055 */
  const size_t u_bytes = chrDstW * sizeof(int32_t);    /* one int32 per pixel */

  unsigned char *line = malloc(bytes);
  CHECK(line, 781);
  int32_t *u = ffc_carve(line, 0, v_off, "chrUPixBuf");
  int32_t *v = ffc_carve(line, v_off, dst_stride, "chrVPixBuf");
  CHECK(v_off + u_bytes <= bytes && v_off + dst_stride <= bytes, 782);

  size_t i = 0, inside = v_off / sizeof(int32_t);
  for (; i < chrDstW && i < inside; i++)
    u[i] = 0x10000 + (int32_t)i;
  if (i < chrDstW) {
    ffc_note(o, line, bytes, u, v_off, &u[i], u_bytes - v_off);
    write_probe_u32((volatile uint32_t *)&u[i], 0x10000 + (uint32_t)i);
    for (i++; i < chrDstW; i++)
      u[i] = 0x10000 + (int32_t)i;
  } else {
    ffc_note(o, line, bytes, u, v_off, &u[chrDstW - 1], sizeof(int32_t));
  }
  for (size_t k = 0; k < chrDstW; k++)
    v[k] = 0x20000 + (int32_t)k;
  o->damage = u[chrDstW - 1] != 0x10000 + (int32_t)(chrDstW - 1); /* U's tail is V's */

  o->defect_text = "U's 256 bytes of int32 samples ran past V's offset of 208 bytes, and V "
                   "then overwrote U's tail";
  o->fixed_text = "the fix places V half a doubled stride on, 416 bytes, past U's 256";
  free(line);
}
