#include "regions.h"
static unsigned char *cursor;
static size_t left;
void wg_regions_init(void *p) {
  cursor = p;
  left = WG_PAYLOAD_BYTES;
}
void wg_region_create(struct wg_region *r, size_t size) {
  if (!size || (size & 15) || size > left)
    wg_fail(202);
  r->alias = cursor;
  r->size = size;
  cursor += size;
  left -= size;
}
void wg_region_renew(struct wg_region *r, unsigned revoke) {
  (void)r;
  (void)revoke;
}
int wg_same_authority(void *a, void *b) { return a == b; }
