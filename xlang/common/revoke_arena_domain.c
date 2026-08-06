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

/* Memory instrumentation getters (this TU owns the all-static rof allocator, so the
 * counters are only reachable here). carved = every byte ever handed out (rof never
 * reclaims arena bytes, so this is the total footprint); peak_slots = high-water of
 * concurrently-live allocations (freed slots are reused, so rof_nslots only grows to
 * the max simultaneous live count -> the working-set proxy); live_* = state at call. */
unsigned long xlang_mem_carved(void) { return rof_carved_total; }
unsigned long xlang_mem_live_bytes(void) { return rof_live_bytes; }
unsigned xlang_mem_live_count(void) { return rof_live_count; }
unsigned xlang_mem_peak_slots(void) { return rof_nslots; }

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

/* Tag-preserving realloc — REQUIRED for Lua (and any real engine that stores
 * capabilities inside GC objects it later grows). This is the SOLE TU allowed to
 * include the all-static rof allocator, so realloc lives here, next to malloc/free.
 *
 * rof never grows in place (each rof_malloc SPLITs a fresh node off the arena top),
 * so realloc ALWAYS moves: carve a new block, copy, revoke the old one. The copy is
 * rof_copy_caps (ldc/stc capability-word moves), NOT a byte loop — a byte loop
 * strips the out-of-band tag off every capability the block holds, so a pointer Lua
 * stored in a realloc'd stack/table would come back untagged and fault on the next
 * deref. Lua's stack and table array-part are realloc'd and MOVED constantly and
 * hold TValues that carry capabilities, so this is load-bearing, not latent.
 *
 * Both blocks are 16-byte aligned and 16-multiple sized (rof_roundup), so we copy
 * min(rof_size(old), rof_size(new)) whole capability words: never a partial tail
 * word (which rof_copy_caps would drop), never past either block's real bounds.
 * rof_free(old) fires the REVOKE — that is what makes stale aliases stop working. */
void *realloc(void *p, size_t size) {
  if (size == 0) {
    rof_free(p);
    return (void *)0;
  }
  void *np = rof_malloc((unsigned long)size);
  if (!np)
    return (void *)0; /* arena depleted: caller (l_alloc) must see NULL, keep old */
  if (p) {
    unsigned long oldsz = rof_size(p);          /* 16-multiple */
    unsigned long newsz = rof_size(np);         /* 16-multiple = roundup(size) */
    unsigned long n = oldsz < newsz ? oldsz : newsz;
    rof_copy_caps(np, p, n);                     /* TAG-PRESERVING (ldc/stc words) */
    rof_free(p);                                 /* the REVOKE of the old node */
  }
  return np;
}
