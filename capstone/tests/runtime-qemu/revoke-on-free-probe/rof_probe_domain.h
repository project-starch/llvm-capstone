#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_REVOKE_ON_FREE_PROBE_DOMAIN_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_REVOKE_ON_FREE_PROBE_DOMAIN_H

/* Domain-side glue: the receive protocol plus the allocator under test. */

#include "revoke_on_free_probe.h"
#include "../../../benchmarks/sqlite/revoke_on_free_alloc.h"

/* The monitor-granted arena, delivered on the func==1 entry and parked in a
 * .bss capability slot until the CALL entry. stc/ldc duplicate rather than move,
 * and cap_compress round-trips the type, so it comes back LIN -- which
 * rof_init/rof_malloc's SPLIT then works on. */
static void *rof_probe_grant;

static inline int rof_probe_receive(void *arg, unsigned func) {
  if (func == ROF_PROBE_DPI_REGION_SHARE) {
    rof_probe_grant = arg;
    return 1;
  }
  return 0;
}

#endif
