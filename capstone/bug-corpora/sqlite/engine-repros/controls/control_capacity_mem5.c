/* CONTROL, not a case: a 20-byte request written at the last byte sqlite3_msize reports, without a
 * realloc, as SQLite does when it grows into a block's unused bytes. No lifetime ends, so the
 * write COMPLETES on the block port, whose bounds are the block. Built by build-virtual.py
 * --observe, against the same engine object as the cases. */
#include "repro322_common.h"

__attribute__((noinline, used)) void control_write(volatile unsigned char *p) { *p = 1; }

static int run_case(void) {
  if (repro_init()) return 1;
  unsigned char *p = sqlite3_malloc(20);
  if (!p) return FAILRC("malloc", 1);
  int block = sqlite3_msize(p);
  control_write(p + block - 1);
  out_text("CONTROL capacity block="); out_uint((unsigned long)block); out_text("\n");
  sqlite3_free(p);
  return 0;
}

REPRO322_MAIN("control_capacity_mem5")
