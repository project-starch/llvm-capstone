#include "corpus.h"

WM_CASE(21) {
/* Row 21 -- tcp.flags.str, fix 69dac89280 ("tcp: simplify tcp.flags.str, fix
 * off-by-one"). SPATIAL.
 *
 * tcp_flags_to_str_first_letter builds a nine-character flag string into a
 * fixed buffer whose size is a local constant:
 *
 *     static const char flags[][4] = { "F","S","R","P","A","U","E","C","N" };
 *     const int maxlength = 16;     // upstream: "Max Flags length"
 *     buf = pbuf = (char *) wmem_alloc(wmem_packet_scope(), maxlength);
 *
 * Nine flags plus the separators and a terminator do not fit in sixteen bytes,
 * and the loop writes them unconditionally. The fix rewrites the function
 * rather than adjusting the constant, which is why this row's two arms differ
 * by the WRITTEN LENGTH rather than by one term -- stated here because every
 * other spatial row in this corpus is a one-term reversal and a reader should
 * not assume this one is.
 *
 * NESTED: the buffer is a packet-scope chunk, so the overrun leaves it and
 * stays inside the block the system allocator handed wmem.
 *
 * WHY THIS ROW IS NOT A DUPLICATE of 14 or 20: there the overrunning index is
 * computed from data -- a payload length, a field extent. Here the buffer is
 * simply too small for a FIXED, compile-time-known amount of output: nine flags
 * into sixteen bytes. No input is needed to reach it, which is also why it was
 * found by inspection rather than by a fuzzer. */
#define MAXLENGTH 16
#define NFLAGS 9
  char *pbuf = wmem_alloc(wm_packet, MAXLENGTH);
  CHECK(pbuf, 1);
  memset(pbuf, 0, MAXLENGTH);
  /* The next chunk of the same block: where the tail of the string lands. */
  unsigned char *successor = wmem_alloc(wm_packet, 64);
  CHECK(successor, 2);
  CHECK((uintptr_t)successor > (uintptr_t)pbuf, 3);
  memset(successor, 0x5a, 64);

  /* What the function writes: one letter per set flag, a separator between
   * them, and a terminator. With all nine set that is 9 + 8 + 1 = 18 bytes. */
  const unsigned written = NFLAGS + (NFLAGS - 1) + 1;
  /* Both halves asserted: the output does not fit, and the overrun stays inside
   * the block rather than leaving it. */
  CHECK(written > MAXLENGTH, 4);
  CHECK((uintptr_t)pbuf + written <= (uintptr_t)successor + 64, 5);

  /* The first byte that does not fit is the labelled crossing. */
  wm_held = (unsigned char *)pbuf + MAXLENGTH;
  wm_mark();
  wm_write_probe(wm_held);
}
