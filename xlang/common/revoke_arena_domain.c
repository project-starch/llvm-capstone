/* Corpus-agnostic revoke-on-free allocator TU for Capstone xlang domains.
 *
 * This is the allocator half of a standalone-shim Capstone domain, extracted so
 * it is tied to NEITHER mruby NOR the Lua-CDP corpus. It provides the ordinary
 * malloc/free (routed through the revoking allocator, so a shim's own free()
 * REVOKES — the whole measurement) plus the arena plumbing the domain driver
 * calls. Any column of standalone shims links exactly this; the parameterised
 * domain driver and host controller are the other two agnostic pieces.
 *
 * The mruby column still carries its own copy of these functions inside
 * mock_mruby_capstone.c (which also holds the mruby VM lifecycle). It can be
 * migrated onto this file the next time that column is re-validated; until then
 * the duplication is deliberate, so touching this file cannot destabilise the
 * already-measured mruby results.
 *
 * revoke_on_free_alloc.h lives under capstone/benchmarks/sqlite/ (add its dir to
 * -I). It is a raw, dependency-free allocator despite the path.
 */
#include "revoke_on_free_alloc.h"
#include <stddef.h>

/* revoke_on_free_alloc.h is entirely `static` — written for a single translation
 * unit. If the domain TU included it too, each TU would get its OWN rof_arena and
 * rof_slots and QEMU would abort in helper_cslcc on the first cap_get_base. THIS
 * TU OWNS THE ALLOCATOR; the domain reaches it only through these functions. */
void xlang_arena_init(void *grant) { rof_init(grant); }
void xlang_set_no_revoke(void) { rof_no_revoke = 1; }

/* Delivery probe (domain XLANG_PROBE_DELIVERY): allocate, revoke, hand the stale
 * alias back. No arithmetic is performed on it, so the next operation is a plain
 * offset-0 load — the route that yields a clean cause-25 fault. */
void *xlang_probe_alloc_and_revoke(void) {
  void *p = rof_malloc(64);
  rof_free(p); /* revokes */
  return p;    /* now-untagged alias */
}

/* Standalone shims allocate through the ordinary names; routing them through the
 * revoking allocator is the whole point — a shim's own free() REVOKES. If these
 * were plain allocations every row would run clean and report MISS while
 * measuring nothing. */
void *malloc(size_t n) { return rof_malloc((unsigned long)n); }
void  free(void *p) { rof_free(p); }
