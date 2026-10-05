#include "corpus.h"

WM_CASE(15) {
/* Row 15 -- OPC UA, fix d24613c461. SPATIAL, LIVE at our v4.6.8 pin, and the
 * only row here that reads BELOW its object rather than past it.
 *
 * verify_padding takes a pointer near the END of the decrypted chunk and a
 * length it reads FROM THE PACKET:
 *
 *   static int verify_padding(const uint8_t *padding)   v4.6.8:opcua.c:324
 *   pad_len = *padding;
 *   for (i = 0; i < pad_len; ++i)
 *       if (padding[-pad_len + i] != pad_len) return -1;          :332
 *
 * `padding` points into `plaintext`, allocated at :642 with
 * wmem_alloc(pinfo->pool, plaintext_len). pad_len is attacker-controlled and
 * can reach 255, so padding[-pad_len] reads an unbounded distance below the
 * chunk. The fix passes an `available` count and rejects pad_len > available;
 * the pin has neither the parameter nor the guard.
 *
 * Note on class: this is class B while the chunk has a predecessor inside the
 * block, and degrades to class A when the chunk is first in a fresh block. The
 * case allocates a predecessor first so the crossing is the chunk bound, which
 * is the shape being claimed -- and asserts it. */
  enum { PLAINTEXT = 32, PAD_LEN = PLAINTEXT + 8 }; /* pad_len > the chunk */
  /* The predecessor the underread lands in, filled with a pattern. */
  unsigned char *before = wmem_alloc(wm_packet, 64);
  CHECK(before, 1);
  memset(before, 0x5a, 64);
  /* :642 -- plaintext = wmem_alloc(pinfo->pool, plaintext_len). */
  unsigned char *plaintext = wmem_alloc(wm_packet, PLAINTEXT);
  CHECK(plaintext, 2);
  memset(plaintext, 0x33, PLAINTEXT);
  /* The predecessor really is below, and adjacent enough that the underread
   * stays inside the one block. Asserted BEFORE the marker. */
  CHECK((uintptr_t)before < (uintptr_t)plaintext, 3);
  /* padding = &plaintext[plaintext_len - *sig_len - 1], i.e. near the end. The
   * sig_len reduction is to 0, so the cursor is the chunk's last byte. */
  unsigned char *padding = plaintext + (PLAINTEXT - 1);
  /* :332 -- padding[-pad_len + i], first iteration i = 0. */
  unsigned char *target = padding - PAD_LEN;
  CHECK((uintptr_t)target < (uintptr_t)plaintext, 4);        /* below the chunk */
  CHECK((uintptr_t)target >= (uintptr_t)before, 5);          /* inside the block */
  wm_held = target;
  wm_mark();
  (void)wm_probe(wm_held);
}
