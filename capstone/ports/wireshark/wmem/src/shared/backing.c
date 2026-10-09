#include "regions.h"
#include "chunks.h"
#include <string.h>
/* WM_DOMAIN here is the FREESTANDING axis, not the capability one: what it
 * decides is whether there is a libc. WM_CAPABILITY decides whether regions
 * carry Sublet handles, and a Capstone process has both. */
#ifndef WM_DOMAIN
#include <stdlib.h>
#endif
/* One region per system request. Released regions of the same size are
 * reissued before fresh payload is carved, as the system allocator would
 * reissue a freed block; nothing is ever returned to the payload. */
struct entry {
  struct wm_region region;
  uintptr_t base;
  size_t size;
  unsigned live;
  /* Held LINEAR for an allocator that carves it (the chunk port), rather than
   * lent as one alias: never reissued by wm_sys_alloc, nor it by them. */
  unsigned linear;
};
static struct entry entries[WM_REGIONS];
static unsigned count, live, peak, temporal;
static uint64_t created;
/* The allocator's own headers, which may not live in memory it hands away. */
static unsigned char *meta;
static size_t meta_used;
void wm_init_backing(void *metadata, void *payload, unsigned mode) {
  if (mode > 1)
    wm_fail(203);
  temporal = mode;
  meta = metadata;
  meta_used = 0;
  wm_regions_init(payload);
  wm_chunks_init(mode);
}
void *wm_meta_alloc(size_t n) {
  n = (n + 15) & ~(size_t)15;
#ifdef WM_DOMAIN
  if (!meta || n > WM_META_BYTES - meta_used)
    wm_fail(212);
  void *p = meta + meta_used;
  meta_used += n;
#else
  void *p = aligned_alloc(16, n);
  if (!p)
    wm_fail(212);
#endif
  memset(p, 0, n);
  return p;
}
static struct entry *find(uintptr_t address, unsigned exact) {
  for (unsigned i = 0; i < count; ++i) {
    struct entry *e = &entries[i];
    if (!e->live || e->linear)
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
    if (!entries[i].live && !entries[i].linear && entries[i].size == rounded)
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
#ifdef WM_CAPABILITY
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
void wm_handback_probe(const void *p) { probe(p); }
/* The chunk port's block: one region of the same size class as wm_sys_alloc's,
 * counted the same way, but handed over LINEAR into *out and never lent. */
size_t wm_block_acquire(size_t n, capstone_cap_slot *out) {
  size_t rounded = (n + 15) & ~(size_t)15;
  struct entry *e = NULL;
  for (unsigned i = 0; i < count && !e; ++i)
    if (!entries[i].live && entries[i].linear && entries[i].size == rounded)
      e = &entries[i];
  if (!e) {
    if (count == WM_REGIONS)
      wm_fail(206);
    e = &entries[count++];
    wm_region_create_linear(&e->region, rounded);
    e->base = capstone_cap_base(&e->region.handle);
    e->size = rounded;
    e->linear = 1;
    ++created;
  }
  capstone_cap_move(&e->region.handle, out);
  e->live = 1;
  if (++live > peak)
    peak = live;
  return e->base;
}
/* The block must come back whole: the caller has revoked its senior handle, so
 * every chunk carved from it is gone and the region is one linear piece again. */
void wm_block_release(size_t base, capstone_cap_slot *in) {
  for (unsigned i = 0; i < count; ++i) {
    struct entry *e = &entries[i];
    if (!e->live || !e->linear || e->base != base)
      continue;
    if (capstone_cap_type(in) != CAPSTONE_CAP_LINEAR ||
        capstone_cap_base(in) != base || capstone_cap_end(in) != base + e->size)
      wm_fail(213);
    capstone_cap_move(in, &e->region.handle);
    e->live = 0;
    --live;
    return;
  }
  wm_fail(211);
}
void *wm_widen(void *p) {
  probe(p);
  struct entry *e = find((uintptr_t)p, 0);
  if (!e)
    wm_fail(210);
  return (char *)e->region.alias + ((uintptr_t)p - e->base);
}
#else
void wm_handback_probe(const void *p) { (void)p; }
void *wm_widen(void *p) { return p; }
#endif
void wm_backing_stats(struct wm_header *out) {
  out->regions_created = created;
  out->regions_peak = peak;
}
