#include "regions.h"
static unsigned char *cursor;
static size_t left;
void wm_regions_init(void *p) {
  cursor = p;
  left = WM_PAYLOAD_BYTES;
}
void wm_region_create(struct wm_region *r, size_t size) {
  if (!size || (size & 15) || size > left)
    wm_fail(202);
  r->alias = cursor;
  r->size = size;
  cursor += size;
  left -= size;
}
void wm_region_renew(struct wm_region *r, unsigned revoke) {
  (void)r;
  (void)revoke;
}
