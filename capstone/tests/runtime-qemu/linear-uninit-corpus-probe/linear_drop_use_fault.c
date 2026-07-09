// row11 go_double_finalize -- LINEAR, double-free. THE MECHANISM.
//
// Stage-1 defect: sqlite3_finalize(stmt) is called twice. The first call frees
// the statement; the second walks freed memory.
//
// Stage-2 shape: `stmt` is a LINEAR (move-only) capability. sqlite3_finalize
// CONSUMES it -- csdrop -- and linearity guarantees no second copy was left
// anywhere for the second call to use. Any later use of the variable therefore
// touches a cleared register, not freed memory.
//
//   stmt = split(arena, mid)   a move-only sub-capability with its own rev node
//   gone = drop(stmt)          sqlite3_finalize() #1: consumes the handle
//   *gone                      sqlite3_finalize() #2: nothing left to use
//
// Expected: FAULT, cause 24 (RISCV_EXCP_UNEXP_OP_TYPE, "Cap mem access requires
// capability"). Cause 24 only says the tag was gone, which several things
// produce, so linear_no_drop_ok runs the identical sequence WITHOUT the drop and
// must succeed.
#include "corpus_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *stmt = corpus_carve_stmt();               /* LINEAR statement handle */
  void *gone = __builtin_capstone_cap_drop(stmt); /* finalize #1: consumed */

  volatile char *p = (volatile char *)gone;
  volatile char v = *p; /* finalize #2: USE AFTER CONSUME -> must FAULT */

  *res = CORPUS_RET_DROP_NOTRAP | (unsigned char)v; /* unreachable */
}
