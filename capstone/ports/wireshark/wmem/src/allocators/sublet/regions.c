#include "regions.h"
static capstone_cap_slot remaining;
static uintptr_t cursor, end;
void wm_regions_init(void *p) {
  capstone_cap_store(&remaining, p);
  cursor = capstone_cap_base(&remaining);
  end = capstone_cap_end(&remaining);
  if (capstone_cap_type(&remaining) != CAPSTONE_CAP_LINEAR ||
      end - cursor != WM_PAYLOAD_BYTES || (cursor & 15))
    wm_fail(201);
}
void wm_region_create(struct wm_region *r, size_t size) {
  if (!size || (size & 15) || size > end - cursor)
    wm_fail(202);
  cursor += size;
  sublet_carve(&remaining, cursor, &r->handle);
  r->size = size;
  r->alias = sublet_take(&r->handle);
}
void wm_region_renew(struct wm_region *r, unsigned revoke) {
  if (revoke) {
    sublet_give(&r->handle);
    r->alias = sublet_take(&r->handle);
  }
}
