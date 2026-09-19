#include "regions.h"
static capstone_cap_slot remaining;
static uintptr_t cursor, end;
void wg_regions_init(void *p) {
  capstone_cap_store(&remaining, p);
  cursor = capstone_cap_base(&remaining);
  end = capstone_cap_end(&remaining);
  if (capstone_cap_type(&remaining) != CAPSTONE_CAP_LINEAR ||
      end - cursor != WG_PAYLOAD_BYTES || (cursor & 15))
    wg_fail(201);
}
void wg_region_create(struct wg_region *r, size_t size) {
  if (!size || (size & 15) || size > end - cursor)
    wg_fail(202);
  cursor += size;
  sublet_carve(&remaining, cursor, &r->handle);
  r->size = size;
  r->alias = sublet_take(&r->handle);
}
void wg_region_renew(struct wg_region *r, unsigned revoke) {
  if (revoke) {
    sublet_give(&r->handle);
    r->alias = sublet_take(&r->handle);
  }
}
int wg_same_authority(void *a, void *b) {
  capstone_cap_slot x, y;
  capstone_cap_store(&x, a);
  capstone_cap_store(&y, b);
  if (capstone_cap_type(&x) != CAPSTONE_CAP_NONLINEAR ||
      capstone_cap_type(&y) != CAPSTONE_CAP_NONLINEAR)
    return 0;
  const volatile uint64_t *xx = (const volatile uint64_t *)&x;
  const volatile uint64_t *yy = (const volatile uint64_t *)&y;
  return xx[0] == yy[0] && xx[1] == yy[1];
}
