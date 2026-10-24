/* Evidence for case 15 (not a gate): the access pattern its trace shows. A 128-byte memsys5 block is
 * read at offset 0x37 through an interior pointer, freed, and read again at the same offset.
 * uaf_interior.c is the same without the read before the free; the pair differs in that one read. */
#include "repro322_common.h"

static int run_case(void) {
  int rc = repro_init();
  if (rc) return rc;
  unsigned char *blk = sqlite3_malloc(100);           /* a 128-byte memsys5 block */
  if (!blk) { out_text("CONTROL-FAILED malloc\n"); return 1; }
  for (int i = 0; i < 100; i++) blk[i] = (unsigned char)i;
  volatile unsigned char *a = blk + 0x37;
  unsigned char before = a[0];
  out_text("ctl read before free value="); out_uint(before); out_text("\n");
  sqlite3_free(blk);
  out_text("ctl freed\n");
  unsigned char after = a[0];
  out_text("ctl stale read value="); out_uint(after); out_text("\n");
  return 0;
}
REPRO322_MAIN("ctlrbf")
