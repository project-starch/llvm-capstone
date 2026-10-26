/* CONTROL, not a case: a realloc that shrinks a 64-byte request to 20 bytes, which memsys5 serves
 * in place (memsys5Realloc's nBytes<=nOld branch), then a read through the old pointer. On the
 * block port every successful realloc is a new lifetime, so the read FAULTS, cause 25 in
 * control_read; with -DSUBLET_REALLOC_KEEPS_LIFETIME (the ablation) it COMPLETES. The line before
 * the read shows the new pointer: same address, contents kept. Built by build-virtual.py
 * --observe, against the same engine object as the cases. */
#include "repro322_common.h"

__attribute__((noinline, used)) unsigned control_read(const volatile unsigned char *p) { return *p; }

static int run_case(void) {
  if (repro_init()) return 1;
  unsigned char *p = sqlite3_malloc(64);
  if (!p) return FAILRC("malloc", 1);
  p[0] = 7;
  unsigned char *q = sqlite3_realloc(p, 20);
  if (!q) return FAILRC("realloc", 1);
  out_text("CONTROL realloc-shrink block="); out_uint((unsigned long)sqlite3_msize(q));
  out_text(" same_address="); out_uint((char *)q - (char *)p == 0);
  out_text(" kept="); out_uint(q[0] == 7);
  out_text("\nCONTROL realloc-shrink mark\n");
  (void)control_read(p);
  sqlite3_free(q);
  return 0;
}

REPRO322_MAIN("control_realloc_shrink_mem5")
