/* The parent, sibling and successor controls of the 2026-10-12 pre-registration
 * (docs/plans/2026-10-12-nested-lifetime-controls-extended.md), on memcached's cache.c object cache:
 * cache_create, cache_alloc, cache_free. The probes, the ACCESS-BEGIN / ACCESS-DONE announcements and
 * the span helpers are the slab controls' own (../lifetime/controls.h, unchanged), so every access is
 * still one load or store in ctl_read_u8 / ctl_write_u8 / ctl_load_ptr, and the stale access is the
 * corpus's labelled read_probe (shared/corpus.h).
 *
 * THE OBJECT LAYOUT, per arm.
 *   virtual-nested-pools (patch 0006, MC_CAPSTONE_SUBLET): cache_create adds one capability slot in
 *     front of each object (bufsize = request + 16). A new object's LIFETIME is CDERIVEd from what
 *     malloc returned, bounded to bufsize; cache_alloc writes the lifetime into the slot and returns
 *     get_object(CDERIVE(lifetime, 0, bufsize)): a child bounded to [slot, slot + bufsize) whose
 *     cursor is the object, slot + 16. cache_free reads the lifetime from the slot through the
 *     caller's pointer, CREVOKEs the caller's object and pushes the LIFETIME on the cache's STAILQ,
 *     whose link then overwrites the slot. So the cache's own reference to a free object's storage is
 *     cache->head.stqh_first, and under patch 0006 that is exactly the CREVOKE parent.
 *   virtual-malloc (the registered release arm, patch 0002 hooks, stock ledger mode 0): no slot; the
 *     object is the virtual-mallocng object and the STAILQ link is written into the object's first
 *     pointer-sized bytes.
 * In both arms the first 16 bytes of the object can hold the free-list link (stock) and the 16 bytes
 * before it the lifetime (protected), so the controls read and write the object's bytes
 * [CC_DATA, CC_DATA + CTL_SPAN) only, which nothing in cache.c touches.
 *
 * WHERE A COMES FROM. cache.c hands out from its own storage (the STAILQ free list) only when the
 * list is not empty; otherwise it mallocs. Every control therefore primes the cache (cache_alloc then
 * cache_free) and takes A with cc_take_from_list, which requires: freecurr drops by one, total does
 * not change (no malloc), A's address is the list head's (plus the slot under patch 0006), and, under
 * patch 0006, A's capability is [slot, slot + bufsize) with its cursor at slot + 16 and the slot,
 * read through A, holds a tagged capability with the same bounds: the shape cache_alloc's patched
 * CDERIVE produces. On the release arm the ledger's object_reuses count rises by one
 * (mcp_object_issue, the free-list branch). */
#ifndef MC_CACHE_CONTROLS_H
#define MC_CACHE_CONTROLS_H
#include "../lifetime/controls.h"
#include <stdlib.h>

#define CC_REQUEST 256u                   /* cache_create's bufsize argument */
#define CC_DATA ((size_t)sizeof(void *)) /* the object bytes past the free-list link */
#ifdef MC_CAPSTONE_SUBLET
#define CC_SLOT ((unsigned long)sizeof(void *)) /* the lifetime slot in front of each object */
#else
#define CC_SLOT 0ul
#endif

__attribute__((unused)) static void cc_state(const cache_t *c, const char *when) {
  printf("CACHE when=%s bufsize=%lu total=%d freecurr=%d limit=%d head=%#lx", when, (unsigned long)c->bufsize,
         c->total, c->freecurr, c->limit, ctl_addr(STAILQ_FIRST(&c->head)));
#ifndef MC_CAPSTONE_SUBLET
  struct mcp_header h;
  mcp_stats(&h);
  printf(" object_reuses=%llu object_releases=%llu", (unsigned long long)h.object_reuses,
         (unsigned long long)h.object_releases);
#endif
  printf("\n");
}

__attribute__((unused)) static void cc_print_cap(const char *tag, const char *name, const void *p) {
#if defined(__CAPSTONE__)
  printf("%s name=%s addr=%#lx base=%#llx end=%#llx tag=%llu\n", tag, name, ctl_addr(p),
         (unsigned long long)__builtin_capstone_cap_get_base((void *)p),
         (unsigned long long)__builtin_capstone_cap_get_end((void *)p),
         (unsigned long long)__builtin_capstone_cap_get_tag((void *)p));
#else
  printf("%s name=%s addr=%#lx\n", tag, name, ctl_addr(p));
#endif
}

/* cache_create, with the patch's bufsize checked: request + 16 under patch 0006, the request without. */
__attribute__((unused)) static cache_t *cc_create(const char *name) {
  cache_t *c = cache_create(name, CC_REQUEST, sizeof(char *));
  CHECK(c, 901);
  cc_state(c, "created");
  CHECK(c->limit == 0 && c->total == 0 && c->freecurr == 0, 902);
  CHECK(c->bufsize == CC_REQUEST + CC_SLOT, 903);
  return c;
}

/* n objects from malloc (the list is empty), then given back in reverse, so the list holds them in
 * allocation order: the first cache_alloc afterwards returns the first one. Their addresses go to
 * out[]. */
__attribute__((unused)) static void cc_prime(cache_t *c, unsigned n, unsigned long *out) {
  void *p[4];
  CHECK(n >= 1 && n <= 4, 904);
  for (unsigned i = 0; i < n; ++i) {
    p[i] = cache_alloc(c);
    CHECK(p[i], 905);
    out[i] = ctl_addr(p[i]);
    printf("PRIME alloc=%u addr=%#lx path=malloc\n", i, out[i]);
  }
  CHECK(c->total == (int)n && c->freecurr == 0, 906);
  for (unsigned i = n; i-- > 0;)
    cache_free(c, p[i]);
  cc_state(c, "primed");
  CHECK(c->freecurr == (int)n && c->total == (int)n, 907);
}

/* The slot in front of a live object, read through the object (as cache_free reads it). Under patch
 * 0006 it is the object's lifetime. Returns it. */
__attribute__((unused)) static void *cc_read_slot_at(const char *name, const char *via, void *obj, const char *file,
                                                     int line) {
#ifdef MC_CAPSTONE_SUBLET
  void *volatile *slot = (void *volatile *)obj - 1;
  printf("ACCESS-BEGIN name=%s kind=read via=%s addr=%#lx site=%s:%d probe=ctl_load_ptr@controls.h:%d\n", name,
         via, ctl_addr(slot), file, line, ctl_load_ptr_line);
  void *life = ctl_load_ptr(slot);
  printf("ACCESS-DONE name=%s value=%#lx\n", name, ctl_addr(life));
  cc_print_cap("LIFETIME", name, life);
  return life;
#else
  (void)name, (void)via, (void)obj, (void)file, (void)line;
  return NULL;
#endif
}
#define cc_read_slot(name, via, obj) cc_read_slot_at(name, via, obj, __FILE__, __LINE__)

/* cache_alloc, required to come from the cache's own free list and, under patch 0006, to have the
 * shape of the patched derive (see the header comment). */
__attribute__((unused)) static void *cc_take_from_list(cache_t *c, const char *name) {
  int total = c->total, freecurr = c->freecurr;
  unsigned long head = ctl_addr(STAILQ_FIRST(&c->head));
  CHECK(freecurr > 0, 910);
#ifndef MC_CAPSTONE_SUBLET
  struct mcp_header before, after;
  mcp_stats(&before);
#endif
  void *obj = cache_alloc(c);
  CHECK(obj, 911);
  unsigned long at = ctl_addr(obj);
  printf("SOURCE name=%s addr=%#lx path=free-list list_head_before=%#lx freecurr=%d->%d total=%d->%d\n", name, at,
         head, freecurr, c->freecurr, total, c->total);
  CHECK(c->freecurr == freecurr - 1 && c->total == total, 912);
  CHECK(at == head + CC_SLOT, 913);
  cc_print_cap("CHUNK-CAP", name, obj);
#ifdef MC_CAPSTONE_SUBLET
  unsigned long base = (unsigned long)__builtin_capstone_cap_get_base(obj);
  unsigned long end = (unsigned long)__builtin_capstone_cap_get_end(obj);
  CHECK(base == head && end - base == c->bufsize, 914);
  void *life = cc_read_slot("slot-through-new-object", name, obj);
  CHECK(__builtin_capstone_cap_get_tag(life) && ctl_addr(life) == head &&
            (unsigned long)__builtin_capstone_cap_get_base(life) == base &&
            (unsigned long)__builtin_capstone_cap_get_end(life) == end, 915);
  printf("CHECK name=%s-patched-derive-shape object=[%#lx,%#lx) cursor=%#lx slot_lifetime=[%#lx,%#lx) ok\n", name,
         base, end, at, base, end);
#else
  mcp_stats(&after);
  printf("CHECK name=%s-ledger-issue object_reuses=%llu->%llu\n", name, (unsigned long long)before.object_reuses,
         (unsigned long long)after.object_reuses);
  CHECK(after.object_reuses == before.object_reuses + 1, 916);
#endif
  return obj;
}

#endif
