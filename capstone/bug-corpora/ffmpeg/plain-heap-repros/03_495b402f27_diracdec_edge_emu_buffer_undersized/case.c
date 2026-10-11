#include "corpus.h"

/* diracdec's edge-emulation buffer, fix 495b402f27. The base is ONE direct
 * allocation -- `s->edge_emu_buffer_base = av_malloc_array(stride, MAX_BLOCKSIZE)`
 * at libavcodec/diracdec.c:342 -- carved into four sub-buffers whose spacing has
 * nothing to do with how much each one is then used for. Here it is the
 * platform's calloc. */

#define MAX_BLOCKSIZE 32 /* diracdec.c:56 */

FFH_CASE(3) {
  /* Case 3 -- diracdec alloc_buffers / dirac_decode_frame_internal, fix
   * 495b402f27. PLAIN HEAP: a one-byte WRITE past the end of a direct
   * av_malloc_array, because the allocation is sized for one sub-buffer and
   * then carved into four.
   *
   * The allocation at the pin (diracdec.c:342):
   *
   *     s->edge_emu_buffer_base = av_malloc_array(stride, MAX_BLOCKSIZE);
   *
   * and the carve (diracdec.c:1898):
   *
   *     for (i = 0; i < 4; i++)
   *         s->edge_emu_buffer[i] = s->edge_emu_buffer_base + i*FFALIGN(p->width, 16);
   *
   * Each of the four sub-buffers is then used by the motion-compensation path
   * for up to MAX_BLOCKSIZE rows of `stride` bytes, but they are spaced only
   * FFALIGN(width, 16) apart and the base holds `stride * MAX_BLOCKSIZE` in
   * total -- enough for ONE of them. So sub-buffer 3 begins at
   * 3*FFALIGN(width,16) and runs off the end. The fix does both halves: it
   * allocates `stride * 4 * MAX_BLOCKSIZE` and respaces the carve to
   * `i * s->buffer_stride * MAX_BLOCKSIZE`.
   *
   * THERE IS NO INNER BOUND TO CROSS HERE, which is why this row is in the
   * plain-heap corpus and not the sub-object one: pre-fix the four sub-buffers
   * OVERLAP each other, so "the sub-buffer's bound" is not a well-defined thing
   * to have crossed. The only bound that exists is the av_malloc_array's own,
   * and the crossing leaves it. */
  const long width = 16;
  const long stride = 16;            /* FFALIGN(width, 16) == width == stride here */
  const long aligned = (width + 15) & ~15L;
  const long base_elems = fixed ? stride * 4 * MAX_BLOCKSIZE : stride * MAX_BLOCKSIZE;
  const long spacing = fixed ? stride * MAX_BLOCKSIZE : aligned;

  unsigned char *base = calloc((size_t)base_elems, 1);
  CHECK(base, 731);
  /* The claim, asserted: at the pin the fourth sub-buffer's last row starts
   * beyond the allocation; under the fix it does not. */
  const long sub = 3;
  const long last_row_off = sub * spacing + (MAX_BLOCKSIZE - 1) * stride;
  CHECK(fixed ? (last_row_off < base_elems) : (last_row_off >= base_elems), 732);

  o->cap = (unsigned long)base_elems;
  o->touched = last_row_off;
  o->crossed = last_row_off >= base_elems;
  o->extent = last_row_off >= base_elems ? last_row_off - base_elems + 1 : 0;

  if (o->crossed) {
    /* Probe only the FIRST crossing byte. The unreduced path writes a whole
     * row, and `extent` above records how far that would run. */
    write_probe_u8(&base[base_elems], 0x41);
    o->damage = 1;
  } else {
    write_probe_u8(&base[last_row_off], 0x41);
    o->damage = 0;
  }

  o->defect_text = "the base was sized stride*MAX_BLOCKSIZE and then carved into FOUR "
                   "sub-buffers spaced FFALIGN(width,16) apart, so sub-buffer 3's last "
                   "row starts past the end of av_malloc_array";
  o->fixed_text = "the fix allocates stride*4*MAX_BLOCKSIZE and respaces the carve to "
                  "i*buffer_stride*MAX_BLOCKSIZE, so all four fit";
  free(base);
}
