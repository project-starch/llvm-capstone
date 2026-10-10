#include "corpus.h"

WM_CASE(20) {
/* Row 20 -- proto_find_undecoded_data, fix ed20250c13 ("proto.c: protect
 * against buffer overflow in proto_find_undecoded_data()"). SPATIAL.
 *
 * The function allocates a bitmap one bit per byte of the frame:
 *
 *     gchar* decoded = (gchar*)wmem_alloc0(wmem_packet_scope(), length / 8 + 1);
 *
 * and then walks every field of the protocol tree marking the bytes it covers:
 *
 *     for (i = fi->start; i < fi->start + fi->length; i++) {
 *         byte = i / 8;
 *         bit = i % 8;
 *         decoded[byte] |= 1 << bit;
 *
 * The loop is bounded by the FIELD's extent and by nothing else. A field whose
 * start plus length reaches past the frame -- which a malformed or
 * reassembled-and-truncated capture produces -- drives `byte` past
 * `length / 8`, and the OR writes into the next chunk.
 *
 * The fix threads the bitmap's own length through in a struct and adds
 * `&& i < decoded->length` to the loop, so the bound is the buffer's rather
 * than the field's.
 *
 * WHAT MAKES THIS ROW DIFFERENT from 14 and 18: there the index is derived from
 * attacker-supplied LENGTHS, and the fix clamps them. Here the index comes from
 * the protocol tree's own field extents, which the dissector itself produced --
 * the loop trusts a sibling subsystem's output rather than an input. The fix
 * does not clamp the field; it bounds the write. That distinction is why the
 * row is here rather than being a duplicate of the clamp cases.
 *
 * NESTED: the bitmap is a packet-scope chunk, so the write leaves it and lands
 * in storage the same block handed out next. */
#define FRAME_LENGTH 64
  const unsigned bitmap_bytes = FRAME_LENGTH / 8 + 1; /* 9 */
  char *decoded = wmem_alloc(wm_packet, bitmap_bytes);
  CHECK(decoded, 1);
  memset(decoded, 0, bitmap_bytes);
  /* The next chunk of the same block: where the overrun lands. */
  unsigned char *successor = wmem_alloc(wm_packet, 64);
  CHECK(successor, 2);
  CHECK((uintptr_t)successor > (uintptr_t)decoded, 3);
  memset(successor, 0x5a, 64);

  /* A field claiming to extend past the frame. fi->start + fi->length is the
   * only thing the loop consults. */
  const unsigned fi_start = FRAME_LENGTH - 8;
  const unsigned fi_length = 32; /* reaches to 88, past the frame's 64 */
  unsigned i = fi_start + fi_length - 1; /* the last index the loop reaches */
  unsigned byte = i / 8;

  /* Both halves asserted: the byte index is past the bitmap, and still inside
   * the block. A reduction whose arithmetic missed would fail here. */
  CHECK(byte >= bitmap_bytes, 4);
  CHECK((uintptr_t)decoded + byte < (uintptr_t)successor + 64, 5);
  /* The fix's guard WOULD have stopped this, which is the other half. */
  CHECK(!(i < bitmap_bytes), 6);

  /* `decoded[byte] |= 1 << bit` -- a read-modify-write on the labelled probe. */
  /* THE FIX, ed20250c13: the bitmap's own length bounds the loop, so its last write is the last
   * bitmap byte. */
  wm_held = (unsigned char *)decoded + (wm_fixed ? bitmap_bytes - 1 : byte);
  wm_mark();
  WM_WRITE_AT(wm_held, decoded, bitmap_bytes);
}
