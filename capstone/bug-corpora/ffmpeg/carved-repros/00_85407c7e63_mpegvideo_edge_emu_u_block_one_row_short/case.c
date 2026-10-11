#include "corpus.h"

/* mpegvideo's edge-emulation scratch, fix 85407c7e63 ("Fix edge emu buffer
 * overlap with interlaced mpeg4"). ONE allocation, sc->edge_emu_buffer --
 * `FF_ALLOCZ_ARRAY_OR_GOTO(avctx, sc->edge_emu_buffer, alloc_size, 4 * 68)` at
 * libavcodec/mpegpicture.c:79 of the fix's parent, alloc_size from :59 -- that
 * mpeg_motion_internal() carves into luma rows, a U block and a V block
 * (libavcodec/mpegvideo_motion.c:328-329):
 *
 *     uint8_t *ubuf = s->sc.edge_emu_buffer + 18 * s->linesize;
 *     uint8_t *vbuf = ubuf + 9 * s->uvlinesize;
 *
 * U gets nine rows, but its edge emulation writes `9 + field_based` of them
 * (:331-335), so for interlaced MPEG-4 U's tenth row IS vbuf's row 0. V's own
 * emulation (:336-340) then overwrites it, and the field MC that reads U at
 * stride 2 * uvlinesize from row `field_select` reads V's samples as U's.
 * The fix carves V one row later, `ubuf + 10 * s->uvlinesize`, and grows the
 * buffer to 4 * 70 rows. */
FFC_CASE(0) {
  const size_t linesize = 64, uvlinesize = 32;            /* a 4:2:0 picture */
  const size_t alloc_size = FFC_ALIGN(linesize + 64, 32); /* mpegpicture.c:59 */
  const size_t nmemb = fixed ? 4 * 70 : 4 * 68;           /* :79, and the fix */
  const size_t bytes = alloc_size * nmemb;
  const int field_based = 1; /* interlaced: the condition the fix's subject names */
  const size_t w = 9, rows = 9 + field_based;             /* :333 and :338 */
  const size_t urows = fixed ? 10 : 9;                    /* :329, and the fix */

  unsigned char *buf = calloc(alloc_size, nmemb);
  CHECK(buf, 701);
  unsigned char *ubuf = ffc_carve(buf, 18 * linesize, urows * uvlinesize, "ubuf");
  unsigned char *vbuf = ffc_carve(buf, 18 * linesize + urows * uvlinesize,
                                  rows * uvlinesize, "vbuf");

  /* U's emulation, rows 0..8: inside U on both arms. */
  for (size_t r = 0; r + 1 < rows; r++)
    memset(ubuf + r * uvlinesize, 0x55, w);
  /* Row 9 is vbuf's row 0 on the buggy arm, and its first byte is the access. */
  unsigned char *last = ubuf + (rows - 1) * uvlinesize;
  ffc_note(o, buf, bytes, ubuf, urows * uvlinesize, last, w);
  write_probe_u8(last, 0x55);
  memset(last + 1, 0x55, w - 1);
  /* V's emulation into its own rows, then the field MC's read of U's row 9. */
  for (size_t r = 0; r < rows; r++)
    memset(vbuf + r * uvlinesize, 0xaa, w);
  o->damage = last[0] != 0x55; /* U's row 9 now holds V's samples */

  o->defect_text = "U's edge emulation wrote 9 + field_based rows into a 9-row block, so its "
                   "tenth row was V's first, which V's emulation then overwrote";
  o->fixed_text = "the fix carves V ten U-rows on, so all ten U rows stay in U";
  free(buf);
}
