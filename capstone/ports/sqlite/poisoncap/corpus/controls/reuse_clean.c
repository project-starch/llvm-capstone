/* Control: ordinary allocate / write / free / reallocate through memsys5, 300 rounds of 16 to 256
 * bytes, no stale access. Must complete on every arm: a fault here is the platform or the harness,
 * and voids the run. Sized to stay inside PoisonCap's quarantine (freed blocks return to memsys5
 * only at a flush, after 4,096 entries or 16 MiB in use): at most 300 blocks of 256 bytes are
 * held, about 75 KB of the 1 MiB heap. Exhausting the heap is their design, not a broken platform,
 * and is read from the cases instead. */
#include "repro322_common.h"

static int run_case(void) {
  int rc = repro_init();
  if (rc) return rc;
  void *keep[8] = {0};
  for (int round = 0; round < 300; round++) {
    int slot = round % 8, size = 16 + (round * 37) % 241;
    sqlite3_free(keep[slot]);
    unsigned char *p = sqlite3_malloc(size);
    if (!p) { out_text("CONTROL-FAILED malloc at round "); out_uint(round); out_text("\n"); return 1; }
    for (int i = 0; i < size; i++) p[i] = (unsigned char)(round + i);
    keep[slot] = p;
  }
  for (int slot = 0; slot < 8; slot++) sqlite3_free(keep[slot]);
  out_text("ctl rounds=300 ok\n");
  return 0;
}
REPRO322_MAIN("ctlreuse")
