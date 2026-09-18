#ifndef SUBLET_SUBLET_H
#define SUBLET_SUBLET_H

#include <capstone/capability.h>

/* Sublet's allocator lifetime operations, built from Capstone slot primitives.
 * take issues an alias while retaining its revocation handle; take_linear
 * lends a region to a nested allocator. give reclaims it for reuse, including
 * initialization when revocation returns an UNINITIALIZED region.
 *
 * Counters belong to each translation unit including this header. A port with
 * several allocator modules must sum their counters when reporting a run.
 * Slot loads/stores and initialization's fill stores are not counted. */
struct sublet_stats {
  unsigned long split, mrev, delin, revoke, init;
};
static struct sublet_stats sublet_stats;

static inline void sublet_split(capstone_cap_slot *lo, unsigned long mid,
                                capstone_cap_slot *hi) {
  capstone_cap_split(lo, mid, hi);
  sublet_stats.split++;
}

/* Keep a senior handle before splitting a region, so one revoke can reclaim
 * all its descendants. The region remains in [slot]. */
static inline void sublet_handle(capstone_cap_slot *slot,
                                 capstone_cap_slot *handle) {
  capstone_cap_make_handle(slot, handle);
  sublet_stats.mrev++;
}

/* [slot] becomes the handle; the caller receives a copyable alias. */
static inline void *sublet_take(capstone_cap_slot *slot) {
  capstone_cap_slot handle;
  capstone_cap_make_handle(slot, &handle);
  void *alias = capstone_cap_delinearize(slot);
  capstone_cap_move(&handle, slot);
  sublet_stats.mrev++;
  sublet_stats.delin++;
  return alias;
}

/* [slot] keeps the handle, [out] receives the linear region. Read its scalar
 * base before moving it. The slots must be distinct. */
static inline unsigned long sublet_take_linear(capstone_cap_slot *slot,
                                               capstone_cap_slot *out) {
  capstone_cap_slot handle;
  capstone_cap_make_handle(slot, &handle);
  unsigned long base = capstone_cap_base(slot);
  capstone_cap_move(slot, out);
  capstone_cap_move(&handle, slot);
  sublet_stats.mrev++;
  return base;
}

/* Reclaim the subtree below [handle] into [slot]. A revoke that kills a
 * linear child returns UNINITIALIZED authority; write it through before INIT
 * makes it readable again. Sublet regions must be aligned and sized to whole
 * capabilities (16 bytes). [handle] is cleared unless it is also [slot]. */
static inline void sublet_give_to(capstone_cap_slot *handle,
                                  capstone_cap_slot *slot) {
  capstone_cap_revoke(handle);
  sublet_stats.revoke++;
  if (capstone_cap_type(handle) == CAPSTONE_CAP_UNINITIALIZED) {
    capstone_cap_initialize_zero(handle);
    sublet_stats.init++;
  }
  capstone_cap_move(handle, slot);
}

static inline void sublet_give(capstone_cap_slot *slot) {
  sublet_give_to(slot, slot);
}

/* Carve the prefix up to end into [to]; any remainder stays in [from]. */
static inline void sublet_carve(capstone_cap_slot *from, unsigned long end,
                                capstone_cap_slot *to) {
  capstone_cap_slot rest;
  if (end < capstone_cap_end(from)) {
    sublet_split(from, end, &rest);
    capstone_cap_move(from, to);
    capstone_cap_move(&rest, from);
  } else {
    capstone_cap_move(from, to);
  }
}

#endif
