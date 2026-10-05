#include "corpus.h"

WM_CASE(16) {
/* Row 16 -- ETSI DCP, fix e8ef9df09d ("ETSI DCP: Fix heap buffer overflow").
 * SPATIAL, and the first row in this corpus whose stale access is a WRITE.
 * Not live at the pin: the fix is in v4.6.8, so this is a fix-reversal.
 *
 * The Reed-Solomon decode works in place and writes the parity bytes at a
 * FIXED offset of 207, while the output cursor advances by rsk, which can be
 * well under 207:
 *
 *   #define PFT_RS_N_MAX 207
 *   #define PFT_RS_P (PFT_RS_K - PFT_RS_N_MAX)                 -- 255-207 = 48
 *   uint8_t *output = (uint8_t*) wmem_alloc(pinfo->pool, decoded_size);  :376
 *   memcpy(output+index_out+PFT_RS_N_MAX, deinterleaved+index_coded,
 *          PFT_RS_P);                                                   :264
 *   index_out += rsk;                                                   :270
 *
 * all at e8ef9df09d^:epan/dissectors/packet-dcp-etsi.c, with decoded_size =
 * fcount*plen (:304). The constants N and K had been swapped relative to
 * RS(255,207); the fix renames them, widens the allocation to
 * decoded_size + PFT_RS_N - rsk, and says in its own comment that the output
 * "does require ... extra space at the end for the parity bytes".
 *
 * Reduced to the first byte the parity copy writes past the buffer. */
  enum { DECODED = 64, PARITY_AT = 207 };
  /* :376 -- the output buffer, sized fcount*plen with no room for parity. */
  unsigned char *output = wmem_alloc(wm_packet, DECODED);
  CHECK(output, 1);
  memset(output, 0, DECODED);
  /* The storage the parity copy overwrites: the next chunk of the same block.
   * Filled with a pattern so the damage is observable, and its position
   * asserted BEFORE the marker. */
  unsigned char *successor = wmem_alloc(wm_packet, 256);
  CHECK(successor, 2);
  memset(successor, 0x5a, 256);
  CHECK((uintptr_t)successor > (uintptr_t)output, 3);
  /* :264 with index_out = 0 -- the copy starts at output + 207, which is past
   * a 64-byte buffer, and 48 bytes of it stay inside the block. */
  CHECK(PARITY_AT >= DECODED, 4);                              /* past the chunk */
  CHECK((uintptr_t)output + PARITY_AT < (uintptr_t)successor + 256, 5); /* in block */
  wm_held = output + PARITY_AT;
  wm_mark();
  wm_write_probe(wm_held);
}
