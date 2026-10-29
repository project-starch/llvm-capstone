/* CONTROL, not a case: a realloc to the size the block already has, which stock SQLite answers in
 * sqlite3Realloc without calling the allocator, then a read through the old pointer. realloc's
 * contract ends the old allocation whether or not the address changes. On the block port
 * (sublet-3220000-memsys5-block.patch) every successful realloc is a new lifetime, so the read
 * FAULTS, cause 25 in control_read; with -DSUBLET_REALLOC_KEEPS_LIFETIME (the ablation) it
 * COMPLETES. The line before the read shows the new pointer: same address, contents kept, the
 * whole block writable. Built by build-virtual.py --observe, against the same engine object as
 * the cases. */
#include "repro322_common.h"

__attribute__((noinline, used)) unsigned control_read(const volatile unsigned char *p) { return *p; }

static int run_case(void) {
  if (repro_init()) return 1;
  unsigned char *p = sqlite3_malloc(20);
  if (!p) return FAILRC("malloc", 1);
  int block = sqlite3_msize(p);
  p[0] = 7;
  unsigned char *q = sqlite3_realloc(p, block);
  if (!q) return FAILRC("realloc", 1);
  q[block - 1] = 1;
  out_text("CONTROL realloc-same block="); out_uint((unsigned long)block);
  out_text(" same_address="); out_uint((char *)q - (char *)p == 0);
  out_text(" kept="); out_uint(q[0] == 7);
  out_text("\nCONTROL realloc-same mark\n");
  (void)control_read(p);
  sqlite3_free(q);
  return 0;
}

REPRO322_MAIN("control_realloc_same_mem5")
