/* CONTROL, not a case: the NON-NESTED counterpart of control_bounds_mem5.c.
 *
 * One 16-byte sqlite3_malloc written 1 KiB past its start, in a build with no
 * memsys5, so the allocation comes from the platform allocator and 1 KiB is well
 * outside it. Under memsys5 the same write lands inside the 256 KiB arena and is
 * invisible; that contrast is why both controls exist.
 *
 * tools/arms.json: on `virtual-mallocng` this must FAULT. A COMPLETE means the
 * arm is not bounding individual allocations, and no spatial silence it reports
 * in this configuration can be read as a miss.
 *
 * Built by build-virtual.py --observe, against the same engine object as the cases. */
#include "repro322_common.h"

__attribute__((noinline, used)) unsigned control_read(const volatile unsigned char *p) { return *p; }
__attribute__((noinline, used)) void control_write(volatile unsigned char *p) { *p = 1; }

static int run_case(void) {
  if (repro_init()) return 1;
  unsigned char *p = sqlite3_malloc(16);
  if (!p) return FAILRC("malloc", 1);
  out_text("CONTROL bounds-malloc mark\n");
  control_write(p + 1024);
  sqlite3_free(p);
  return 0;
}

REPRO322_MAIN("control_bounds_malloc")
