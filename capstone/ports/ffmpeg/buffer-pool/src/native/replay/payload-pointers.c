/* Native functional control: ordinary pointers, without capability revocation.
 */
#include "payload-backend.h"

static unsigned char *remaining;

void ff2_payload_init(void *p, size_t n) {
  remaining = p;
  ff2_pool_init_region((uintptr_t)p, n);
}

void *ff2_payload_issue_pointer(struct payload_block *b, unsigned mode) {
  (void)mode;
  return b->full_alias;
}

int ff2_payload_same_authority(const void *a, const void *b) { return a == b; }

void ff2_payload_carve(struct payload_block *b, size_t offset) {
  b->region.c = remaining + offset;
}

void ff2_payload_prepare_backing(struct payload_block *b, unsigned mode) {
  if (mode != 2 || !b->full_alias)
    b->full_alias = b->region.c;
}

void ff2_payload_return_lease(struct payload_block *b, unsigned mode) {
  (void)b;
  (void)mode;
}

void ff2_payload_free_backing(struct payload_block *b, unsigned mode) {
  (void)b;
  (void)mode;
}

void ff2_payload_report_stats(struct ff2_header *report) { (void)report; }
