/* Capstone mode 2: fresh authority for each lease, revoked on pool return. */
#include "pool-leases.h"
#include "payload-backend.h"
#include <sublet/sublet.h>

static size_t init_bytes;

void *ff2_sublet_issue(struct payload_block *b) {
  return sublet_take(&b->region);
}
void ff2_sublet_return(struct payload_block *b) {
  unsigned long before = sublet_stats.init;
  sublet_give(&b->region);
  if (sublet_stats.init != before)
    init_bytes += b->rounded;
  b->alias = NULL;
}

void ff2_sublet_add_stats(struct ff2_header *h) {
  h->split += sublet_stats.split;
  h->mrev += sublet_stats.mrev;
  h->delin += sublet_stats.delin;
  h->revoke += sublet_stats.revoke;
  h->init += sublet_stats.init;
  h->init_bytes += init_bytes;
}
