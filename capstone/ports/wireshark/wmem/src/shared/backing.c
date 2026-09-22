#include "regions.h"
#include <string.h>
/* One region per system request. Released regions of the same size are
 * reissued before fresh payload is carved, as the system allocator would
 * reissue a freed block; nothing is ever returned to the payload. */
struct entry {
  struct wm_region region;
  uintptr_t base;
  size_t size;
  unsigned live;
};
static struct entry entries[WM_REGIONS];
static unsigned count, live, peak, temporal;
static uint64_t created;
void wm_init_backing(void *metadata, void *payload, unsigned mode) {
  (void)metadata;
  if (mode > 1)
    wm_fail(203);
  temporal = mode;
  wm_regions_init(payload);
}
static struct entry *find(uintptr_t address, unsigned exact) {
  for (unsigned i = 0; i < count; ++i) {
    struct entry *e = &entries[i];
    if (!e->live)
      continue;
    if (exact ? address == e->base
              : address >= e->base && address < e->base + e->size)
      return e;
  }
  return NULL;
}
static void *issue(struct entry *e) {
  e->live = 1;
  if (++live > peak)
    peak = live;
  return e->region.alias;
}
void *wm_sys_alloc(size_t n) {
  if (!n || n > WM_PAYLOAD_BYTES || n > SIZE_MAX - 15)
    wm_fail(205);
  size_t rounded = (n + 15) & ~(size_t)15;
  for (unsigned i = 0; i < count; ++i)
    if (!entries[i].live && entries[i].size == rounded)
      return issue(&entries[i]);
  if (count == WM_REGIONS)
    wm_fail(206);
  struct entry *e = &entries[count++];
  wm_region_create(&e->region, rounded);
  e->base = (uintptr_t)e->region.alias;
  e->size = rounded;
  ++created;
  return issue(e);
}
void wm_sys_free(void *p) {
  if (!p)
    return;
  struct entry *e = find((uintptr_t)p, 1);
  if (!e)
    wm_fail(207);
  wm_region_renew(&e->region, temporal);
  e->live = 0;
  --live;
}
void *wm_sys_realloc(void *p, size_t n) {
  if (!p)
    return wm_sys_alloc(n);
  if (!n) {
    wm_sys_free(p);
    return NULL;
  }
  struct entry *e = find((uintptr_t)p, 1);
  if (!e)
    wm_fail(208);
  if (((n + 15) & ~(size_t)15) <= e->size)
    return e->region.alias;
  void *q = wm_sys_alloc(n);
  memcpy(q, e->region.alias, e->size);
  wm_sys_free(p);
  return q;
}
void *wm_epoch(void *p) {
  struct entry *e = find((uintptr_t)p, 0);
  if (!e)
    wm_fail(209);
  wm_region_renew(&e->region, temporal);
  return (char *)e->region.alias + ((uintptr_t)p - e->base);
}
#ifdef WM_DOMAIN
/* A stale pointer handed back to the allocator must fail here, on its own
 * authority, before any block-wide authority is looked up by address. The
 * label lets the fault oracle name this access. */
__attribute__((noinline)) static void probe(const volatile unsigned char *p) {
  unsigned long x;
  __asm__ volatile(".globl wm_widen_probe\nwm_widen_probe:\nlbu %0, 0(%1)\n"
                   : "=r"(x)
                   : "r"(p)
                   : "memory");
  (void)x;
}
void *wm_widen(void *p) {
  probe(p);
  struct entry *e = find((uintptr_t)p, 0);
  if (!e)
    wm_fail(210);
  return (char *)e->region.alias + ((uintptr_t)p - e->base);
}
#else
void *wm_widen(void *p) { return p; }
#endif
void wm_backing_stats(struct wm_header *out) {
  out->regions_created = created;
  out->regions_peak = peak;
}
