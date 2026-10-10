/* CONTROL, not a case: one 16-byte SQLite allocation through sqlite3_malloc -- the
 * allocator under every case -- written 1 KiB past its start: past any memsys5 block
 * a 16-byte request can be rounded to, and still inside the 256 KiB arena.
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
  unsigned char *p = sqlite3_malloc(16);
  if (!p) return FAILRC("malloc", 1);
  out_text("CONTROL bounds-mem5 mark\n");
  control_write(p + 1024);
  sqlite3_free(p);
  return 0;
}

REPRO322_MAIN("control_bounds_mem5")
