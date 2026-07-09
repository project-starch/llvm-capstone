// SCAFFOLD probe 1/2 (task-007 step 3): the SQLite lifecycle, in miniature.
//
//   arena  = <monitor linear grant>          // sqlite3_config(SQLITE_CONFIG_HEAP)
//   buf    = arena_carve(256)                // the statement's column-name buffer
//   fill buf.alias; read it back             // sqlite3_column_name(); caller uses it
//   arena_revoke(&buf)                       // sqlite3_finalize()
//   read buf.alias again                     // the caller's CACHED handle -> FAULT
//
// This is the row3 "after" in the shape the SQLite integration will take: the
// protected value is a SPLIT sub-capability with its own revocation node, so
// revoking it at finalize does not touch the rest of the heap (see
// held_arena_survives_revoke.c for that half).
//
// Expected: fault, cause 25 at -O1/-O2 (alias register-held: tag intact, node
// revoked -- self-proving), cause 24 at -O0 (alias spilled, so the reload clears
// the tag). held_no_revoke_ok.c is the control for the cause-24 case.
#include "probe_linear_arena.h"

#define LIFECYCLE_CARVE_LEN 256UL

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  probe_arena_init(probe_arena);
  probe_protected_buf buf = probe_arena_carve(LIFECYCLE_CARVE_LEN);

  volatile char *value = (volatile char *)buf.alias; /* handed to the "caller" */
  value[PROBE_OFFSET] = (char)PROBE_SENTINEL_LIVE;
  volatile char live = value[PROBE_OFFSET]; /* the caller reads it: ok */
  (void)live;

  probe_arena_revoke(&buf); /* sqlite3_finalize() */

  volatile char v = value[PROBE_OFFSET]; /* STALE HANDLE -> must FAULT */

  *res = PROBE_RET_LIFECYCLE_NOTRAP | (unsigned char)v; /* unreached if it faults */
}
