#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_HIER_REVOKE_PROBE_DOMAIN_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_HIER_REVOKE_PROBE_DOMAIN_H

/* Domain-side glue: the receive protocol plus the hierarchical allocator under
 * test. The child allocations here are carved with the FLAT allocator's
 * rof_malloc (pointed at a connection's sub-arena via hier_activate), exactly as
 * SQLite's xMalloc will be in the real row7 domain -- so this probe exercises
 * the same derivation chain, not a hand-rolled shortcut. */

#include "hier_revoke_probe.h"
#include "../../../benchmarks/sqlite/revoke_on_free_hier_alloc.h"

static void *hier_probe_grant;

static inline int hier_probe_receive(void *arg, unsigned func) {
  if (func == HIER_PROBE_DPI_REGION_SHARE) {
    hier_probe_grant = arg;
    return 1;
  }
  return 0;
}

/* Open a connection's sub-arena and carve one 64-byte child from it via the
 * flat allocator, returning the child alias. Mirrors "open connection, allocate
 * its first object" -- the child is a SPLIT descendant of c->rev. */
static inline void *hier_probe_open_with_child(hconn *c) {
  hier_open(c, HIER_PROBE_SUBARENA);
  hier_activate(c);
  void *child = rof_malloc(64);
  hier_deactivate(c);
  return child;
}

#endif
