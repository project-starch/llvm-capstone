#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_LINEAR_UNINIT_CORPUS_DOMAIN_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_LINEAR_UNINIT_CORPUS_DOMAIN_H

/* Domain-side glue for the row11/row14 corpus probes.
 *
 * The receive protocol -- how a domain gets hold of the monitor's REGION_SHARE
 * capability -- is not restated here. It is proven, documented and implemented
 * once, in the held-cap probe, and pulled in below:
 *
 *     ../intra-domain-mrev-revoke-probe/probe_domain.h
 *
 * From it these probes take `probe_arena` (the parked grant), `probe_receive()`
 * (the REGION_SHARE entry) and `probe_cap_split()` (the raw cssplit encoding).
 * Duplicating any of that would let the two copies drift, and the receive
 * protocol is the single subtlest thing in either probe.
 *
 * What the two rows need on top of the grant:
 *
 *   row14 (UNINIT). csrevoke leaves its handle typed by whether the revoked
 *   junior run was still linear -- cap_rev_tree_revoke() returns retain_data,
 *   and op_helper.c:710 turns that into LIN (data retained) or UNINIT (data not
 *   retained, cursor parked at end). So the ONLY thing separating an UNINIT
 *   capability from the LIN one the held-cap probe gets is the csdelin: revoke a
 *   still-linear lineage and the retained handle comes back UNINIT. No monitor
 *   op, no start.S change. corpus_uninit_handle() below is that derivation.
 *
 *   row11 (LINEAR). A move-only sub-capability, carved with cssplit so it has a
 *   revocation node of its own, consumed by csdrop. corpus_carve_stmt() below.
 */

#include "../intra-domain-mrev-revoke-probe/probe_domain.h"
#include "linear_uninit_corpus_probe.h"

/* row14: turn the linear grant into an UNINIT capability over the same bytes.
 *
 *   R = mrev(arena)   -- a node senior to the arena's; the arena keeps working
 *   revoke(R)         -- sweeps the junior run, which still contains the arena's
 *                        LINEAR node, so data cannot be retained: the returned
 *                        handle is CAP_TYPE_UNINIT with cursor == end
 *
 * Deliberately NO csdelin on the arena. Delinearising it first is exactly what
 * the held-cap probe does, and it is why that probe's revoke yields LIN. The
 * grant is consumed: `probe_arena` and every copy of it now name an invalidated
 * revocation node. That is the point -- this models a freshly allocated
 * connection wrapper whose `sqlite3 *db` is not usable yet. */
static inline void *corpus_uninit_handle(void) {
  void *arena = probe_arena;
  void *rev = __builtin_capstone_cap_mrev(arena);
  return __builtin_capstone_cap_revoke(rev);
}

/* row11: carve the "statement handle" out of the arena. cssplit gives the high
 * half a FRESH revocation node, so dropping it says nothing about the low half.
 * The low half stays in `probe_arena` as the surviving "connection". */
static inline void *corpus_carve_stmt(void) {
  void *arena = probe_arena;
  unsigned long base = __builtin_capstone_cap_get_base(arena);
  void *stmt = probe_cap_split(&arena, base + CORPUS_SPLIT_MID);
  probe_arena = arena; /* [base, base + CORPUS_SPLIT_MID) */
  return stmt;         /* LINEAR, [base + CORPUS_SPLIT_MID, base + size) */
}

#endif
