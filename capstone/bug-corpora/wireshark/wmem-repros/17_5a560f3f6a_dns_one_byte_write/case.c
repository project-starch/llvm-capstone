#include "corpus.h"

WM_CASE(17) {
/* Row 17 -- DNS, fix 5a560f3f6a ("dns: fix off-by-one buffer overflow (write)").
 * SPATIAL, a WRITE, and the smallest crossing in the corpus: exactly one byte.
 * Not live at the pin -- a fix-reversal.
 *
 * expand_dns_name builds the printable name into a packet-scope buffer of
 * maxname bytes and then hands g_snprintf a size of maxname + 1, so the
 * terminator can land one byte past the chunk:
 *
 *   np = (guchar *)wmem_alloc(wmem_packet_scope(), maxname);
 *   print_len = g_snprintf(np, maxname + 1, "\\[x");
 *   print_len = g_snprintf(np, maxname + 1, "%02x", ...);
 *   print_len = g_snprintf(np, maxname + 1, "/%d]", bit_count);
 *
 * at 5a560f3f6a^:epan/dissectors/packet-dns.c. The fix is the same three calls
 * with maxname instead of maxname + 1. Liveness note: the pin reads
 * snprintf(np, maxname, ...) at :1677, :1689 and :1703 -- the fixed form under
 * a renamed function, which is why the triage script's added-line probe called
 * this one live and the pinned source refutes it.
 *
 * Reduced to the single byte at index maxname, which is the one a size of
 * maxname + 1 permits and a size of maxname does not. */
  enum { MAXNAME = 32 };
  /* The chunk the write leaves by one byte. */
  unsigned char *np = wmem_alloc(wm_packet, MAXNAME);
  CHECK(np, 1);
  memset(np, 0, MAXNAME);
  /* The storage that one byte lands in: the next chunk of the same block.
   * Position asserted BEFORE the marker. */
  unsigned char *successor = wmem_alloc(wm_packet, 64);
  CHECK(successor, 2);
  memset(successor, 0x5a, 64);
  CHECK((uintptr_t)successor > (uintptr_t)np, 3);
  /* g_snprintf with a size of MAXNAME + 1 may write index MAXNAME: one past. */
  CHECK((uintptr_t)np + MAXNAME < (uintptr_t)successor + 64, 4);
  wm_held = np + MAXNAME;
  wm_mark();
  wm_write_probe(wm_held);
}
