/* Spatially bounded leases from the shared reusable arena. */
#include "payload-backend.h"
#include <cheri/cheric.h>

#ifndef __CHERI_PURE_CAPABILITY__
#error "This backend requires CHERI purecap"
#endif

static unsigned char *arena;
static size_t bounds_slack, max_bounds_slack, leases;

void ff2_payload_init(void *p, size_t n) {
  arena = p;
  ff2_pool_init_region((uintptr_t)p, n);
}
void ff2_payload_carve(struct payload_block *b, size_t offset) {
  b->region.c = arena + offset;
}
void ff2_payload_prepare_backing(struct payload_block *b, unsigned mode) {
  if (mode != 0)
    ff2_fail(320); /* Refuse to label spatial execution as temporal. */
  b->full_alias = cheri_setbounds(b->region.c, b->requested);
  if (!cheri_gettag(b->full_alias) ||
      cheri_getbase(b->full_alias) != cheri_getaddress(b->region.c) ||
      cheri_getlen(b->full_alias) < b->requested ||
      cheri_getlen(b->full_alias) > b->rounded)
    ff2_fail(321);
}
void *ff2_payload_issue_pointer(struct payload_block *b, unsigned mode) {
  if (mode != 0)
    ff2_fail(320);
  size_t slack = cheri_getlen(b->full_alias) - b->requested;
  bounds_slack += slack;
  if (slack > max_bounds_slack)
    max_bounds_slack = slack;
  leases++;
  return b->full_alias;
}
int ff2_payload_same_authority(const void *a, const void *b) {
  return __builtin_cheri_equal_exact(a, b);
}
void ff2_payload_return_lease(struct payload_block *b, unsigned mode) {
  (void)b;
  (void)mode;
}
void ff2_payload_free_backing(struct payload_block *b, unsigned mode) {
  (void)b;
  (void)mode;
}
void ff2_payload_report_stats(struct ff2_header *report) {
  /* Extension fields; existing event wire format and mode 0 stay unchanged. */
  report->reserved[0] = leases;
  report->reserved[1] = bounds_slack;
  report->reserved[2] = max_bounds_slack;
  report->reserved[3] = sizeof(void *);
}
