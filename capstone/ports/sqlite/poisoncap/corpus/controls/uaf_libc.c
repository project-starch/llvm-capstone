/* Evidence, not a gate: a planted use-after-free of one C library malloc block, outside memsys5.
 * With the C library's revocation on, their libc quarantines and poisons the freed block, so the
 * stale read is expected to fault; with it off, the block is simply free and the read returns.
 * Recorded to show what the C library's layer protects; no corpus case's object comes from it. */
#include "repro322_common.h"
#include <stdlib.h>

static int run_case(void) {
  volatile unsigned char *p = malloc(96);
  if (!p) { out_text("CONTROL-FAILED malloc\n"); return 1; }
  for (int i = 0; i < 96; i++) p[i] = (unsigned char)(0xA0 + i);
  free((void *)p);
  out_text("ctl freed\n");
  unsigned char seen = p[5];          /* the stale read */
  out_text("ctl stale read value="); out_uint(seen); out_text("\n");
  return 0;
}
REPRO322_MAIN("ctllibcuaf")
