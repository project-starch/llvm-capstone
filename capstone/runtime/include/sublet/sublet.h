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
#ifdef SUBLET_R43_LCC_PROBE
  /* R-43 arm discriminator, OFF unless SUBLET_R43_LCC_PROBE is defined. Reads the LCC node
   * validity of &sublet_stats.split immediately before the counter access that trapped cause 25
   * on R-42, and makes that access data-dependent on the read so the LSU cannot issue ahead of it
   * (R-45: issue is not held behind an in-flight DYN op). `split += __live` keeps the counter
   * identical when every probe reads live, so a completed run still prints split=5568 and any
   * shortfall is a dead reading. rs2 is the SELECTOR field for lcc (literal x0 = selector 0) and a
   * REGISTER for cincoffset -- not an "r" operand, which would encode a register number. */
  { unsigned long __live; unsigned long *__p = &sublet_stats.split;
    __asm__ volatile(
      ".insn r 0x5b, 0x1, 0x04, %0, %1, x0\n"   /* lcc        __live, __p, sel 0 (node validity) */
      "and   t3, %0, x0\n"                      /* t3 = 0, but dependent on __live              */
      ".insn r 0x5b, 0x1, 0x0c, %1, %1, t3\n"   /* cincoffset __p, __p, t3 -- __p awaits the lcc */
      : "=&r"(__live), "+r"(__p) : : "t3", "memory");
    *__p += __live; }
#else
  sublet_stats.split++;
#endif
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
