#ifndef WG_REGIONS_H
#define WG_REGIONS_H
#include "port.h"
#ifdef WG_DOMAIN
#include <sublet/sublet.h>
struct wg_region {
  capstone_cap_slot handle;
  void *alias;
  size_t size;
};
#else
struct wg_region {
  void *alias;
  size_t size;
};
#endif
void wg_regions_init(void *payload);
void wg_region_create(struct wg_region *r, size_t size);
void wg_region_renew(struct wg_region *r, unsigned revoke);
int wg_same_authority(void *a, void *b);
#endif
