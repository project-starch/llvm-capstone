#ifndef WM_REGIONS_H
#define WM_REGIONS_H
#include "port.h"
#ifdef WM_CAPABILITY
#include <sublet/sublet.h>
struct wm_region {
  capstone_cap_slot handle;
  void *alias;
  size_t size;
};
#else
struct wm_region {
  void *alias;
  size_t size;
};
#endif
void wm_regions_init(void *payload);
void wm_region_create(struct wm_region *r, size_t size);
void wm_region_renew(struct wm_region *r, unsigned revoke);
#ifdef WM_CAPABILITY
/* A region left LINEAR in r->handle and not lent (the chunk port's blocks). */
void wm_region_create_linear(struct wm_region *r, size_t size);
size_t wm_block_acquire(size_t n, capstone_cap_slot *out);
void wm_block_release(size_t base, capstone_cap_slot *in);
/* This translation unit's sublet counters: a report must sum them with the
 * chunk port's, since sublet.h counts per translation unit. */
void wm_region_counts(uint64_t *revokes, uint64_t *inits);
#endif
#endif
