/* CONTROL, not a case: a realloc larger than the whole heap, which memsys5 cannot serve. A failed
 * realloc leaves the old allocation as it was, so the read through the old pointer COMPLETES on
 * every configuration, the block port included. Built by build-virtual.py --observe, against the
 * same engine object as the cases. */
#include "repro322_common.h"

__attribute__((noinline, used)) unsigned control_read(const volatile unsigned char *p) { return *p; }

static int run_case(void) {
  if (repro_init()) return 1;
  unsigned char *p = sqlite3_malloc(20);
  if (!p) return FAILRC("malloc", 1);
  p[0] = 7;
  unsigned char *q = sqlite3_realloc(p, 0x20000000);
  out_text("CONTROL realloc-fail failed="); out_uint(q == 0);
  out_text(" kept="); out_uint(control_read(p) == 7);
  out_text("\n");
  sqlite3_free(q ? q : p);
  return 0;
}

REPRO322_MAIN("control_realloc_fail_mem5")
