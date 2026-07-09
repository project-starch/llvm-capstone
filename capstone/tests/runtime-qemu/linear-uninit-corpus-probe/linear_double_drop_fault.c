// row11, the literal shape: sqlite3_finalize(stmt) called twice in a row.
//
// linear_drop_use_fault traps on a later DEREFERENCE of the consumed handle.
// This traps on the second FINALIZE itself, which is what the Go binding
// actually does -- it never dereferences the statement, it hands it back to
// SQLite a second time. csdrop raises "Unexpected operand type (24)" when its
// operand is not a capability, so the double-finalize faults at the call, before
// any freed memory is reachable.
//
// The log line distinguishes it from linear_drop_use_fault: both raise cause 24,
// but this one prints "DROP requires capability" (op_helper.c helper_csdrop) and
// that one prints "Cap mem access requires capability" (_helper_access_with_cap).
//
// Expected: FAULT, cause 24, at the second drop.
#include "corpus_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *stmt = corpus_carve_stmt();

  stmt = __builtin_capstone_cap_drop(stmt); /* finalize #1: ok, consumes it */
  stmt = __builtin_capstone_cap_drop(stmt); /* finalize #2: must FAULT */

  *res = CORPUS_RET_DOUBLE_DROP_NOTRAP; /* unreachable */
  (void)stmt;
}
