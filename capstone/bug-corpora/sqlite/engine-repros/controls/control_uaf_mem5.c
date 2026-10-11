/* CONTROL, not a case: one SQLite allocation through sqlite3_malloc -- the
 * allocator under every case -- freed, then read through its alias.
 * tools/arms.json says what it must do on each configuration: memsys5 stock
 * hands out offsets inside one arena, so it COMPLETES; memsys5 on its Sublet
 * port issues each block as its own capability and retires it on free, so it
 * FAULTS, in the probe below. Built by build-virtual.py --observe, against the
 * same engine object as the cases. */
#include "repro322_common.h"

__attribute__((noinline, used)) unsigned control_read(const volatile unsigned char *p) { return *p; }
__attribute__((noinline, used)) void control_write(volatile unsigned char *p) { *p = 1; }

static int run_case(void) {
  if (repro_init()) return 1;
  unsigned char *p = sqlite3_malloc(64);
  if (!p) return FAILRC("malloc", 1);
  p[0] = 7;
  sqlite3_free(p);
  out_text("CONTROL uaf-mem5 mark\n");
  (void)control_read(p);
  return 0;
}

REPRO322_MAIN("control_uaf_mem5")
