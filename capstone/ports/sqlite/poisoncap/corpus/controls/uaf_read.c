/* Control: a planted use-after-free of one memsys5 block, through the same harness, heap and
 * flags as the corpus cases. On PoisonCap's published memsys5 the read must fault (poison:
 * SIGPROT, PROT_CHERI_SEALED, and "poison exception" on the console); on the spatial arm (their
 * NESTED_TEMPORAL off) the block is simply free again and the read returns. Prints the value it
 * read so a completed run shows what the stale pointer saw. */
#include "repro322_common.h"

static int run_case(void) {
  int rc = repro_init();
  if (rc) return rc;
  volatile unsigned char *p = sqlite3_malloc(96);
  if (!p) { out_text("CONTROL-FAILED malloc\n"); return 1; }
  for (int i = 0; i < 96; i++) p[i] = (unsigned char)(0xA0 + i);
  sqlite3_free((void *)p);
  out_text("ctl freed\n");
  unsigned char seen = p[5];          /* the stale read */
  out_text("ctl stale read value="); out_uint(seen); out_text("\n");
  return 0;
}
REPRO322_MAIN("ctluaf")
