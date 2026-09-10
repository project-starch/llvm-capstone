#ifndef CAPSTONE_BENCHMARKS_SQLITE_REVOKE_ON_FREE_HIER_ALLOC_H
#define CAPSTONE_BENCHMARKS_SQLITE_REVOKE_ON_FREE_HIER_ALLOC_H

/* A HIERARCHICAL revoke-on-free allocator (row7 fork -- HIERARCHICAL-REVOKE).
 *
 * The flat revoke-on-free allocator (revoke_on_free_alloc.h) carves every
 * allocation as an independent SPLIT off one arena: allocations are SIBLINGS
 * with independent revocation nodes, so freeing one says nothing about another.
 * That is the right shape for BORROW-REVOKE (row 3) and LINEAR (row 11), where
 * one handle's lifetime is independent. It is the WRONG shape for
 * HIERARCHICAL-REVOKE (rows 4/5/7/8/9/10/12), where destroying a PARENT
 * (a connection / its host wrapper) must revoke every CHILD (its statements /
 * cursors) as one cascade.
 *
 * This layer adds per-connection SUB-ARENAS on top of the flat primitives:
 *
 *   - hier_open(c, size): carve a sub-arena for connection `c` off the main
 *     arena (a fresh SPLIT node), then MREV it. The MREV handle `c->rev` is
 *     SENIOR to the sub-arena's node, so it is senior to EVERYTHING later
 *     derived from that sub-arena.
 *   - while `c` is the active connection, the flat allocator carves SQLite's
 *     allocations (the sqlite3 object, the Vdbe statement, everything) as SPLIT
 *     children of `c`'s sub-arena -- i.e. DESCENDANTS of c->rev.
 *   - hier_close(c): REVOKE c->rev. Because every child is a descendant of that
 *     senior node, the whole sub-lineage -- the connection AND its live
 *     statements -- is swept in one operation. A sibling connection `d`, whose
 *     sub-arena is a different SPLIT off the main arena, is NOT a descendant of
 *     c->rev, so it SURVIVES. That scoping is what makes it HIERARCHICAL and not
 *     a global heap wipe.
 *
 * WHY THE REVOKE IS FIRED AT close, NOT BY SQLite FREEING MEMORY: SQLite's
 * sqlite3_close_v2 is a ZOMBIE close -- with a live (un-finalized) statement it
 * marks the connection dead but does NOT free the statement's memory (it defers
 * until the statement is finalized). So there is no xFree of the statement at
 * close for the flat allocator to turn into a revoke. The cascade therefore
 * comes from the capability tree this layer builds (revoke c->rev at the
 * wrapper's teardown), exactly modelling a binding whose connection wrapper owns
 * its statement wrappers and tears the whole subtree down. This is intra-domain:
 * SPLIT + MREV + REVOKE only, no monitor/region/start.S change.
 *
 * ATTRIBUTION: SQLite's xMalloc(n) is context-free -- it does not know which
 * connection an allocation belongs to. So the driver marks which connection is
 * active (hier_activate) around each connection's open/prepare/step calls; the
 * flat allocator then carves from that connection's sub-arena. The driver knows
 * the hierarchy (it opened the connection and prepared the statement on it), so
 * this attribution is honest, not a bolt-on.
 */

#include "revoke_on_free_alloc.h"

typedef struct {
  void *arena; /* LIN sub-arena, shrinks as children are carved */
  void *rev;   /* senior MREV handle; revoke sweeps this sub-arena's subtree */
  int live;
} hconn;

/* The top-level grant remainder, from which each connection's sub-arena is
 * carved. Distinct from rof_arena, which this layer repoints per connection. */
static void *hier_main_arena;

static inline void hier_init(void *grant) { hier_main_arena = grant; }

/* Carve a `size`-byte sub-arena off the main arena for a new connection and
 * MREV it. sub stays LIN (mrev retains its source), so the flat allocator can
 * SPLIT children off it; c->rev is senior to sub and to every such child. */
static inline void hier_open(hconn *c, unsigned long size) {
  void *m = hier_main_arena;
  unsigned long end = __builtin_capstone_cap_get_end(m);
  void *sub = rof_split(&m, end - size); /* sub=[end-size,end), fresh node, LIN */
  hier_main_arena = m;
  c->arena = sub;
  c->rev = __builtin_capstone_cap_mrev(sub); /* senior to sub's whole subtree */
  c->live = 1;
}

/* Point the flat allocator at connection c's sub-arena, so subsequent SQLite
 * allocations become SPLIT children (descendants of c->rev). Call before this
 * connection's open/prepare/step; pair with hier_deactivate after. */
static inline void hier_activate(hconn *c) { rof_arena = c->arena; }

/* Save back the (now shrunk) sub-arena after allocating on connection c. */
static inline void hier_deactivate(hconn *c) { c->arena = rof_arena; }

/* Parent teardown: revoke the whole sub-lineage (connection + live statements).
 * Models the connection wrapper's destruction at sqlite3_close_v2. */
static inline void hier_close(hconn *c) {
  if (c->live) {
    __builtin_capstone_cap_revoke(c->rev);
    c->live = 0;
  }
}

#endif
