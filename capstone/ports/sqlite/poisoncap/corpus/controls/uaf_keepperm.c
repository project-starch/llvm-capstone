/* Evidence, not a gate: uaf_read with the heap handed to SQLite as the bare static array, which
 * keeps CHERI_PERM_POISON, instead of through repro_heap_view(). Their QEMU skips the poison check
 * for a capability with that permission (cheri-helper-utils.h:391), so on the published arm this
 * stale read is expected to return. Recorded to show what the harness's permission strip is for;
 * the corpus cases never take this path. */
#include "repro322_common.h"

static int run_case(void) {
  int rc = sqlite3_config(SQLITE_CONFIG_HEAP, sqlite_heap_storage, (int)sizeof sqlite_heap_storage, 64);
  if (rc != SQLITE_OK) { out_text("CONTROL-FAILED config-heap\n"); return rc; }
  out_text("repro heap_perm_poison=");
  out_uint((cheri_getperm(sqlite_heap_storage) & CHERI_PERM_POISON) != 0);
  out_text("\n");
  if ((rc = sqlite3_initialize()) != SQLITE_OK) { out_text("CONTROL-FAILED initialize\n"); return rc; }
  volatile unsigned char *p = sqlite3_malloc(96);
  if (!p) { out_text("CONTROL-FAILED malloc\n"); return 1; }
  for (int i = 0; i < 96; i++) p[i] = (unsigned char)(0xA0 + i);
  sqlite3_free((void *)p);
  out_text("ctl freed\n");
  unsigned char seen = p[5];
  out_text("ctl stale read value="); out_uint(seen); out_text("\n");
  return 0;
}
REPRO322_MAIN("ctlkeepperm")
