/* Evidence for case 15 (not a gate): uaf_read_before_free.c without the read before the free. */
#include "repro322_common.h"

static int run_case(void) {
  int rc = repro_init();
  if (rc) return rc;
  unsigned char *blk = sqlite3_malloc(100);           /* a 128-byte memsys5 block */
  if (!blk) { out_text("CONTROL-FAILED malloc\n"); return 1; }
  for (int i = 0; i < 100; i++) blk[i] = (unsigned char)i;
  volatile unsigned char *a = blk + 0x37;
  sqlite3_free(blk);
  out_text("ctl freed\n");
  unsigned char after = a[0];
  out_text("ctl stale read value="); out_uint(after); out_text("\n");
  return 0;
}
REPRO322_MAIN("ctlinterior")
