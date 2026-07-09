#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_INTRA_DOMAIN_MREV_REVOKE_LINEAR_ARENA_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_INTRA_DOMAIN_MREV_REVOKE_LINEAR_ARENA_H

/* SCAFFOLD for the row3 Option B SQLite integration (task-007 step 3).
 *
 * A monitor-granted linear arena, out of which the domain carves INDEPENDENTLY
 * REVOCABLE sub-buffers. This is the shape SQLite needs: one protected value per
 * statement (e.g. the column-name buffer), handed to the caller as an ordinary
 * pointer, revoked at sqlite3_finalize(), after which the caller's cached copy
 * faults -- while the rest of the heap keeps working.
 *
 * WHY THIS AND NOT "MREV SQLite's own heap pointer" -- read before integrating:
 *
 *   memsys5 hands out allocations as `&mem5.zPool[i * szAtom]`, which lowers to
 *   `cincoffset`. helper_cscincoffset does `*rd_v = *rs1_v` and then only bumps
 *   bounds.cursor: the derived capability inherits the pool's rev_node_id, its
 *   type, AND its bounds. So a memsys5 allocation is not a distinct capability
 *   in any sense the revocation tree can see:
 *
 *     - MREV of one allocation mints a node senior to the POOL's node, so the
 *       revoke would sweep the entire heap, not that allocation;
 *     - and it cannot even get that far: the pool must be delinearised before
 *       SQLite may copy it around (`mem5.zPool = zByte` is a movc, and a LINEAR
 *       capability is consumed by copy -- task-005 finding C3), so allocations
 *       come out NONLIN, and helper_csmrev *asserts* CAP_TYPE_LIN.
 *
 *   SPLIT is the only derivation that mints a fresh revocation node
 *   (cap_rev_tree_split); SHRINK/SHRINKTO copy rev_node_id unchanged. But the
 *   emulator has NO merge/unsplit op, so splitting is one-way -- an allocator
 *   that coalesces freed neighbours, as memsys5 does, cannot be built on it.
 *
 * Hence: point memsys5 at a granted arena if you like (it works -- memsys5 never
 * does inttoptr, only pointer arithmetic and pointer differences off zPool), but
 * do NOT expect its allocations to be revocable. Carve the values you actually
 * want to protect out of a SEPARATE linear arena with this helper.
 *
 * See README.md, "Step 3: memsys5 feasibility".
 */

#include "probe_domain.h"

/* One independently revocable buffer carved out of the arena. */
typedef struct {
  void *alias; /* NONLIN, hand this to the consumer */
  void *rev;   /* REV handle: the owner keeps it, fires it at the lifecycle point */
} probe_protected_buf;

/* The LINEAR remainder of the grant. Parked in a .bss capability slot: stc/ldc
 * duplicate rather than move, so the remainder stays LIN and stays carve-able. */
static void *probe_arena_cursor;

static inline void probe_arena_init(void *grant) { probe_arena_cursor = grant; }

/* Carve `len` bytes off the TAIL of the remaining arena.
 *
 * One-way: the arena only ever shrinks (no merge op). `len` must be strictly
 * less than what remains -- cssplit asserts base < mid < end.
 */
static inline probe_protected_buf probe_arena_carve(unsigned long len) {
  probe_protected_buf b;
  void *cur = probe_arena_cursor; /* LIN */
  unsigned long end = __builtin_capstone_cap_get_end(cur);

  void *hi = probe_cap_split(&cur, end - len); /* fresh rev node, still LIN */
  probe_arena_cursor = cur;                    /* [base, end - len) */

  b.rev = __builtin_capstone_cap_mrev(hi);   /* senior to hi's node only */
  b.alias = __builtin_capstone_cap_delin(hi); /* copyable, still revocable */
  return b;
}

/* The lifecycle point (sqlite3_finalize). Every alias derived from b->alias --
 * including copies the consumer cached -- stops dereferencing. */
static inline void probe_arena_revoke(probe_protected_buf *b) {
  __builtin_capstone_cap_revoke(b->rev);
}

#endif
