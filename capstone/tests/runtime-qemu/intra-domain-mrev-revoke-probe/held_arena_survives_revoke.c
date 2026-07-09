// SCAFFOLD probe 2/2 (task-007 step 3): revoking one protected value must not
// take the heap down with it.
//
// Two statements are live at once. Finalizing the first revokes only its own
// buffer: the second statement's buffer, and the un-carved remainder of the
// arena (where an allocator would keep serving), both keep working.
//
//   a = arena_carve(256)      // statement 1's column-name buffer
//   b = arena_carve(256)      // statement 2's
//   write through a and b, and through the arena remainder
//   arena_revoke(&a)          // sqlite3_finalize(stmt1)
//   b and the remainder must still dereference
//
// This is what makes SPLIT the right derivation: cap_rev_tree_split gives each
// carve a fresh revocation node, so the three lineages are siblings and the sweep
// of a's node reaches neither b nor the remainder. cincoffset/shrink would have
// shared one node and revoking any of them would have swept all three.
//
// Expected: no fault, retval 0x22370077.
#include "probe_linear_arena.h"

#define SURVIVE_CARVE_LEN 256UL

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  probe_arena_init(probe_arena);
  probe_protected_buf a = probe_arena_carve(SURVIVE_CARVE_LEN);
  probe_protected_buf b = probe_arena_carve(SURVIVE_CARVE_LEN);

  volatile char *va = (volatile char *)a.alias;
  volatile char *vb = (volatile char *)b.alias;
  volatile char *rest = (volatile char *)probe_arena_cursor; /* un-carved */

  va[PROBE_OFFSET] = 0x11;
  vb[PROBE_OFFSET] = 0x22;
  rest[PROBE_OFFSET] = 0x44;

  probe_arena_revoke(&a); /* sqlite3_finalize(stmt1) -- a's lineage only */

  /* Siblings, not descendants: untouched by the sweep. */
  volatile char sb = vb[PROBE_OFFSET];
  volatile char sr = rest[PROBE_OFFSET];

  if ((unsigned char)sb != 0x22 || (unsigned char)sr != 0x44) {
    *res = (PROBE_RET_ARENA_SURVIVES_OK & ~0xffu) | 0xbbu; /* wrong bytes */
    return;
  }
  *res = PROBE_RET_ARENA_SURVIVES_OK; /* 0x22370077 */
}
