#ifndef WM_REGIONS_H
#define WM_REGIONS_H
#include "port.h"
#ifdef WM_DOMAIN
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
#endif
