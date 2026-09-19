#ifndef FFPOOL_PAYLOAD_BACKEND_H
#define FFPOOL_PAYLOAD_BACKEND_H

#include "trace.h"

#include <capstone/capability-slot.h>

/* The pool owns these records. Backends operate on capability slots in place;
 * a record must never be copied because its region and outer slots may be
 * linear. The pool manages allocation/reuse bookkeeping and metadata; the
 * selected native or Capstone backend manages payload authority. */
struct payload_block {
  capstone_cap_slot region, outer;
  void *full_alias, *alias, *meta;
  uintptr_t address;
  size_t requested, rounded;
  unsigned alive, idle;
#ifdef FFPOOL_POISONCAP
  unsigned long poison_epoch;
#endif
};

/* Called by ff2_payload_init after the backend has stored the incoming grant.
 * Passing the scalar address here avoids forwarding a linear capability through
 * another C call. Only the backend retains the remaining region's authority. */
void ff2_pool_init_region(uintptr_t base, size_t capacity);

/* CMake links exactly one implementation of these operations. */
void ff2_payload_carve(struct payload_block *block, size_t offset);
void *ff2_payload_issue_pointer(struct payload_block *block, unsigned mode);
int ff2_payload_same_authority(const void *a, const void *b);
void ff2_payload_prepare_backing(struct payload_block *block, unsigned mode);
void ff2_payload_return_lease(struct payload_block *block, unsigned mode);
void ff2_payload_free_backing(struct payload_block *block, unsigned mode);
void ff2_payload_report_stats(struct ff2_header *report);

#endif
