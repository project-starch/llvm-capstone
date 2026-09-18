#ifndef CAPSTONE_CAPABILITY_H
#define CAPSTONE_CAPABILITY_H

#include <capstone/capability-slot.h>

/* Capstone-only slot operations, with no allocator state or counters.
 * Linear values stay inside asm: load, operate, then store before returning
 * to C. Only scalar metadata and delinearized aliases are returned by value.
 * Destination slots must be empty or contain authority already revoked.
 *
 * Opcode 0x5b: ldc funct3=3, stc funct3=4, other operations funct3=1.
 * LCC selectors: x1 type, x2 cursor, x3 base, x4 end.
 */
enum {
  CAPSTONE_CAP_LINEAR = 0,
  CAPSTONE_CAP_NONLINEAR = 1,
  CAPSTONE_CAP_UNINITIALIZED = 3,
  CAPSTONE_CAP_EMPTY = 7
};

/* Split [lo] at mid. Keep the lower part in [lo], put the upper in [hi].
 * The slots must be distinct. */
static inline void capstone_cap_split(capstone_cap_slot *lo, unsigned long mid,
                                      capstone_cap_slot *hi) {
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%0)\n"
                   ".insn r 0x5b, 0x1, 0x06, t1, t0, %1\n"
                   ".insn s 0x5b, 0x4, t0, 0(%0)\n"
                   ".insn s 0x5b, 0x4, t1, 0(%2)\n"
                   :
                   : "r"(lo), "r"(mid), "r"(hi)
                   : "t0", "t1", "memory");
}

/* Create a revocation handle senior to [slot], preserving its region.
 * The slots must be distinct. A handle created before a split covers both
 * halves and all descendants subsequently derived from them. */
static inline void capstone_cap_make_handle(capstone_cap_slot *slot,
                                            capstone_cap_slot *handle) {
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%0)\n"
                   ".insn r 0x5b, 0x1, 0x08, t1, t0, x0\n"
                   ".insn s 0x5b, 0x4, t1, 0(%1)\n"
                   ".insn s 0x5b, 0x4, t0, 0(%0)\n"
                   :
                   : "r"(slot), "r"(handle)
                   : "t0", "t1", "memory");
}

/* Consume [slot] and return a copyable alias to its region. */
static inline void *capstone_cap_delinearize(capstone_cap_slot *slot) {
  void *alias;
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(%1)\n"
                   ".insn s 0x5b, 0x4, x0, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x03, %0, x0, x0\n"
                   : "=&r"(alias)
                   : "r"(slot)
                   : "memory");
  return alias;
}

/* Revoke descendants. The handle becomes a LINEAR or UNINITIALIZED region;
 * this operation does not initialize or otherwise write that region. */
static inline void capstone_cap_revoke(capstone_cap_slot *slot) {
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%0)\n"
                   ".insn r 0x5b, 0x1, 0x00, x0, t0, x0\n"
                   ".insn s 0x5b, 0x4, t0, 0(%0)\n"
                   :
                   : "r"(slot)
                   : "t0", "memory");
}

/* Initialize an UNINITIALIZED region by writing null capabilities from its
 * cursor to its end, then executing INIT. Its bounds and cursor must be
 * capability-aligned (16 bytes). A trailing partial capability makes INIT
 * trap; the loop never spins on that tail. Stores advance the UNINIT cursor.
 * This is a composite helper, not a single instruction or an implicit part
 * of revoke: callers choose whether and when initialization is required. */
static inline void capstone_cap_initialize_zero(capstone_cap_slot *slot) {
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%0)\n"
                   ".insn r 0x5b, 0x1, 0x04, t1, t0, x2\n"
                   ".insn r 0x5b, 0x1, 0x04, t2, t0, x4\n"
                   "1: addi t3, t1, 16\n"
                   "bgtu t3, t2, 2f\n"
                   ".insn s 0x5b, 0x4, x0, 0(t0)\n"
                   "mv t1, t3\n"
                   "j 1b\n"
                   "2: .insn r 0x5b, 0x1, 0x09, t0, t0, x0\n"
                   ".insn s 0x5b, 0x4, t0, 0(%0)\n"
                   :
                   : "r"(slot)
                   : "t0", "t1", "t2", "t3", "memory");
}

/* Move authority between slots and clear its old location. Identical slots
 * are permitted. */
static inline void capstone_cap_move(capstone_cap_slot *from,
                                     capstone_cap_slot *to) {
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%0)\n"
                   ".insn s 0x5b, 0x4, x0, 0(%0)\n"
                   ".insn s 0x5b, 0x4, t0, 0(%1)\n"
                   :
                   : "r"(from), "r"(to)
                   : "t0", "memory");
}

/* Capture an incoming grant immediately, before another C copy can consume
 * it. Keep this operation inline even in unoptimized builds. */
static inline __attribute__((always_inline)) void
capstone_cap_store(capstone_cap_slot *slot, void *cap) {
  __asm__ volatile(".insn s 0x5b, 0x4, %1, 0(%0)\n"
                   :
                   : "r"(slot), "r"(cap)
                   : "memory");
}

/* Metadata readers restore the capability to its slot after LCC: even
 * reading a linear capability out of memory can move it. */
static inline unsigned long capstone_cap_type(capstone_cap_slot *slot) {
  unsigned long v;
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x04, %0, t0, x1\n"
                   ".insn s 0x5b, 0x4, t0, 0(%1)\n"
                   : "=&r"(v)
                   : "r"(slot)
                   : "t0", "memory");
  return v;
}

static inline unsigned long capstone_cap_base(capstone_cap_slot *slot) {
  unsigned long v;
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x04, %0, t0, x3\n"
                   ".insn s 0x5b, 0x4, t0, 0(%1)\n"
                   : "=&r"(v)
                   : "r"(slot)
                   : "t0", "memory");
  return v;
}

static inline unsigned long capstone_cap_end(capstone_cap_slot *slot) {
  unsigned long v;
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x04, %0, t0, x4\n"
                   ".insn s 0x5b, 0x4, t0, 0(%1)\n"
                   : "=&r"(v)
                   : "r"(slot)
                   : "t0", "memory");
  return v;
}

/* Clear storage whose authority is dead or has moved; this does not revoke. */
static inline void capstone_cap_clear(capstone_cap_slot *slot) {
  __asm__ volatile(".insn s 0x5b, 0x4, x0, 0(%0)\n" : : "r"(slot) : "memory");
}

#endif
