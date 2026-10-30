/* CONTROL, not a case: the NON-NESTED counterpart of control_uaf_mem5.c.
 *
 * Same allocation call the cases use -- sqlite3_malloc -- but in a build with no
 * memsys5, so repro_init() never calls SQLITE_CONFIG_HEAP and SQLite's allocator
 * IS the platform's: musl mallocng compiled for Capstone on the virtual profile,
 * CheriBSD's malloc on the purecap guest. There is no arena and no sub-allocation:
 * the object the defect is about is the object the system allocator handed out.
 *
 * tools/arms.json: on `virtual-mallocng` this must FAULT. That is the whole point
 * of the non-nested column -- the same mechanisms that are blind inside memsys5
 * see this immediately, because the lifetime that ended is one the allocator knows
 * about. A COMPLETE here means the arm has no per-object lifetime at all and every
 * silence it reports in this configuration is worthless.
 *
 * Built by build-virtual.py --observe, against the same engine object as the cases. */
#include "repro322_common.h"

__attribute__((noinline, used)) unsigned control_read(const volatile unsigned char *p) { return *p; }
__attribute__((noinline, used)) void control_write(volatile unsigned char *p) { *p = 1; }

static int run_case(void) {
  if (repro_init()) return 1;
  unsigned char *p = sqlite3_malloc(64);
  if (!p) return FAILRC("malloc", 1);
  p[0] = 7;
  sqlite3_free(p);
  out_text("CONTROL uaf-malloc mark\n");
  (void)control_read(p);
  return 0;
}

REPRO322_MAIN("control_uaf_malloc")
