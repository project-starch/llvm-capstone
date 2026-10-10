#include "corpus.h"

/* rv34's B-frame prediction scratch, fix d2213b6493 ("Fix buffer size used for
 * MC of B frames after a resolution change"). ONE allocation carved at the
 * linesize of the moment (libavcodec/rv34.c of the fix's parent):
 *
 *     r->tmp_b_block_base = av_malloc(s->linesize * 48);                      // :1315
 *     r->tmp_b_block_y[i] = r->tmp_b_block_base + i * 16 * s->linesize;       // :1317
 *     r->tmp_b_block_uv[i] = r->tmp_b_block_base + 32 * s->linesize + ...;    // :1319-1320
 *
 * The re-carve is guarded by `s->width != r->si.width || ...` (:1311), but the
 * size-change branch just above has already set `s->width = r->si.width`
 * (:1294-1295), so after a resolution increase the guard is dead and the
 * regions stay carved at the OLD linesize. Weighted bidirectional MC then
 * writes each direction's 16 rows at the NEW linesize (:789):
 *
 *     Y = r->tmp_b_block_y [dir]     +  xoff     +  yoff    *s->linesize;
 *
 * With the linesize doubled, direction 0's rows 8..15 ARE direction 1's rows
 * 0..7; direction 1's prediction then overwrites them, and rv4_weight()
 * averages direction 1 with itself. The fix frees the block on a size change
 * and re-carves it unconditionally when it is absent.
 *
 * Up to a doubling everything stays inside the 48 old rows; a larger jump
 * escapes. */
FFC_CASE(10) {
  const size_t ls_old = 64, ls_new = 128; /* a resolution change that doubles the stride */
  const size_t ls_carve = fixed ? ls_new : ls_old;
  const size_t bytes = ls_carve * 48;     /* :1315 */

  unsigned char *base = malloc(bytes);
  CHECK(base, 801);
  unsigned char *y[2];
  y[0] = ffc_carve(base, 0, 16 * ls_carve, "tmp_b_block_y[0]");
  y[1] = ffc_carve(base, 16 * ls_carve, 16 * ls_carve, "tmp_b_block_y[1]");
  (void)ffc_carve(base, 32 * ls_carve, 16 * ls_carve, "tmp_b_block_uv");
  CHECK(16 * ls_carve + 15 * ls_new + 16 <= bytes, 802); /* direction 1 stays inside */

  /* Direction 0's 16x16 prediction at the new linesize (xoff = yoff = 0). */
  const size_t region = 16 * ls_carve;
  size_t r = 0;
  for (; r < 16 && r * ls_new + 16 <= region; r++)
    memset(y[0] + r * ls_new, 0xa0, 16);
  if (r < 16) {
    unsigned char *at = y[0] + r * ls_new;
    ffc_note(o, base, bytes, y[0], region, at, (size_t)(y[0] + 15 * ls_new + 16 - at));
    write_probe_u8(at, 0xa0);
    memset(at + 1, 0xa0, 15);
    for (r++; r < 16; r++)
      memset(y[0] + r * ls_new, 0xa0, 16);
  } else {
    ffc_note(o, base, bytes, y[0], region, y[0] + 15 * ls_new + 15, 1);
  }
  /* Direction 1, then rv4_weight's read of direction 0's row 8. */
  for (r = 0; r < 16; r++)
    memset(y[1] + r * ls_new, 0xb0, 16);
  o->damage = y[0][8 * ls_new] != 0xa0;

  o->defect_text = "after the linesize doubled, direction 0's prediction ran from "
                   "tmp_b_block_y[0] into tmp_b_block_y[1], which direction 1 then overwrote";
  o->fixed_text = "the fix re-carves the block at the new linesize";
  free(base);
}
