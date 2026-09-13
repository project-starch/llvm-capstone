/* Sublet: the capability discipline for a nested allocator, as operations on slots.
 *
 * An allocator under Sublet never holds a capability in a C variable. A linear
 * capability moves when it is copied (movc nulls its source), so a compiler that
 * copies freely turns it into null where the code did not expect it. Every
 * capability an allocator owns therefore lives in a sublet_cap slot in memory,
 * and every operation below loads it, works on it in fixed temporaries, and
 * stores the result back before C sees anything. What C does see is the alias:
 * the delinearised, copyable capability that names an object for the caller.
 *
 * The recipe (nested-allocators-paper, experiments/A7-ports.md):
 *
 *   block from the level below   sublet_take_linear: mrev, the block stays LIN
 *   carve an object              sublet_split, then sublet_take: mrev, delin
 *   free an object               sublet_give: revoke the object's handle; the
 *                                slot holds the region again, linear, and the
 *                                next sublet_take hands it out under a new handle
 *   merge, reset, destroy        sublet_give_to on the handle senior to the
 *                                children: one revoke, whatever hangs below dies
 *
 * A handle is what mrev returns: a REV capability senior to the region's node.
 * revoke walks the junior run of nodes, so a handle taken before a split covers
 * both halves, and a handle taken by the level below covers a whole sub-pool.
 * After revoke the handle is the region again: LIN if only aliases hung below,
 * UNINIT if a linear child did, and init is what turns the latter into LIN.
 * init succeeds only on a region that has been written through: revoke leaves
 * the cursor at the base, a capability-grained store at the cursor advances it
 * by one capability, and init asks for a cursor at the end. So sublet_give_to
 * fills the region before it inits, and a merge and a pool's destruction cost
 * a write of the block. That write is the reclaim: the borrower's bytes are
 * gone before anyone can read the region again.
 *
 * Encodings: opcode 0x5b, funct3 1, funct7 revoke 0 lcc 4 split 6 mrev 8 init 9
 * delin 3; ldc funct3 3, stc funct3 4 (capstone-qemu insn32.decode). The lcc
 * selector rides in the rs2 field: x1 type, x2 cursor, x3 base, x4 end.
 */
#ifndef SUBLET_H
#define SUBLET_H

typedef struct sublet_cap {
  void *c;
} sublet_cap;

/* how often each primitive ran, per translation unit that includes this header */
struct sublet_stats {
  unsigned long split, mrev, delin, revoke, init;
};
static struct sublet_stats sublet_stats;

/* [lo] = [base, mid), keeps its node; [hi] = [mid, end), a fresh node */
static inline void sublet_split(sublet_cap *lo, unsigned long mid, sublet_cap *hi) {
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%0)\n"
                   ".insn r 0x5b, 0x1, 0x06, t1, t0, %1\n"
                   ".insn s 0x5b, 0x4, t0, 0(%0)\n"
                   ".insn s 0x5b, 0x4, t1, 0(%2)\n"
                   :
                   : "r"(lo), "r"(mid), "r"(hi)
                   : "t0", "t1", "memory");
  sublet_stats.split++;
}

/* the object goes out: [slot] keeps the handle, the caller gets the alias */
static inline void *sublet_take(sublet_cap *slot) {
  void *alias;
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x08, t1, %0, x0\n"
                   ".insn s 0x5b, 0x4, t1, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x03, %0, x0, x0\n"
                   : "=&r"(alias)
                   : "r"(slot)
                   : "t1", "memory");
  sublet_stats.mrev++;
  sublet_stats.delin++;
  return alias;
}

/* the block goes out linear, to a level that carves it: [slot] keeps the handle,
   [out] gets the block. Returns the block's base address. */
static inline unsigned long sublet_take_linear(sublet_cap *slot, sublet_cap *out) {
  unsigned long base;
  /* the base is read before the store: a store of a linear capability nulls
     its source register on hardware that enforces linearity */
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x08, t1, t0, x0\n"
                   ".insn s 0x5b, 0x4, t1, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x04, %0, t0, x3\n"
                   ".insn s 0x5b, 0x4, t0, 0(%2)\n"
                   : "=&r"(base)
                   : "r"(slot), "r"(out)
                   : "t0", "t1", "memory");
  sublet_stats.mrev++;
  return base;
}

/* a handle senior to [slot]'s region, into [handle]; [slot] keeps the region.
   Taken before a split, it merges the halves again with one revoke. */
static inline void sublet_handle(sublet_cap *slot, sublet_cap *handle) {
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%0)\n"
                   ".insn r 0x5b, 0x1, 0x08, t1, t0, x0\n"
                   ".insn s 0x5b, 0x4, t1, 0(%1)\n"
                   ".insn s 0x5b, 0x4, t0, 0(%0)\n"
                   :
                   : "r"(slot), "r"(handle)
                   : "t0", "t1", "memory");
  sublet_stats.mrev++;
}

/* revoke the handle in [handle]: every capability derived below it dies, and
   [slot] holds the region again, linear, ready for the next take. [handle] is
   cleared when it is another slot.

   A revoke that killed a linear child hands the region back uninitialised, and
   init is refused until the region has been written through. The loop is that
   write: a null capability at the cursor, which advances it, from wherever the
   revoke left it to the end. An emulator that leaves the cursor at the end
   instead (capstone-qemu before Q-07, the build the recorded passes ran on)
   runs no iteration of it, and this is the single init it was there.

   The fill is deliberately not counted. It is what init costs on a machine
   that follows the text, not a primitive of the discipline, and the counts a
   pass records are the same on both emulators; the cycles are not, and are the
   board's to say.

   A region that is not a whole number of capabilities long cannot be written
   through: the loop stops at the last whole one and the init behind it traps,
   which is a fault at a named pc rather than a spin. memsys5 hands out powers
   of two of its 64-byte atom and the pool is carved from those, so no region
   here is such a one. */
static inline void sublet_give_to(sublet_cap *handle, sublet_cap *slot) {
  unsigned long inited;
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x00, x0, t0, x0\n"
                   ".insn r 0x5b, 0x1, 0x04, %0, t0, x1\n"
                   "addi %0, %0, -3\n"
                   "bnez %0, 3f\n"
                   ".insn r 0x5b, 0x1, 0x04, t1, t0, x2\n"
                   ".insn r 0x5b, 0x1, 0x04, t2, t0, x4\n"
                   "1: addi t3, t1, 16\n"
                   "bgtu t3, t2, 2f\n"
                   ".insn s 0x5b, 0x4, x0, 0(t0)\n"
                   "mv t1, t3\n"
                   "j 1b\n"
                   "2: .insn r 0x5b, 0x1, 0x09, t0, t0, x0\n"
                   "li %0, 1\n"
                   "j 4f\n"
                   "3: li %0, 0\n"
                   "4: .insn s 0x5b, 0x4, x0, 0(%1)\n"
                   ".insn s 0x5b, 0x4, t0, 0(%2)\n"
                   : "=&r"(inited)
                   : "r"(handle), "r"(slot)
                   : "t0", "t1", "t2", "t3", "memory");
  sublet_stats.revoke++;
  sublet_stats.init += inited;
}

/* free an object or a sub-pool: its handle and its region share one slot */
static inline void sublet_give(sublet_cap *slot) { sublet_give_to(slot, slot); }

/* [to] = [from], [from] cleared: a linear capability changes slots */
static inline void sublet_move(sublet_cap *from, sublet_cap *to) {
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%0)\n"
                   ".insn s 0x5b, 0x4, x0, 0(%0)\n"
                   ".insn s 0x5b, 0x4, t0, 0(%1)\n"
                   :
                   : "r"(from), "r"(to)
                   : "t0", "memory");
}

/* a capability that arrived in a register (a grant from the level below or the
   monitor) into its slot, without passing through a C variable again */
static inline void sublet_store(sublet_cap *slot, void *cap) {
  __asm__ volatile(".insn s 0x5b, 0x4, %1, 0(%0)\n" : : "r"(slot), "r"(cap) : "memory");
}

/* the type and bounds of the capability in [slot]; type 7 means the slot is
   empty. A load of a linear capability moves it out of memory on hardware that
   enforces linearity, so every reader stores it back. */
static inline unsigned long sublet_type(sublet_cap *slot) {
  unsigned long v;
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x04, %0, t0, x1\n"
                   ".insn s 0x5b, 0x4, t0, 0(%1)\n"
                   : "=&r"(v) : "r"(slot) : "t0", "memory");
  return v;
}
static inline unsigned long sublet_base(sublet_cap *slot) {
  unsigned long v;
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x04, %0, t0, x3\n"
                   ".insn s 0x5b, 0x4, t0, 0(%1)\n"
                   : "=&r"(v) : "r"(slot) : "t0", "memory");
  return v;
}
static inline unsigned long sublet_end(sublet_cap *slot) {
  unsigned long v;
  __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%1)\n"
                   ".insn r 0x5b, 0x1, 0x04, %0, t0, x4\n"
                   ".insn s 0x5b, 0x4, t0, 0(%1)\n"
                   : "=&r"(v) : "r"(slot) : "t0", "memory");
  return v;
}

/* [to] = [from]'s region up to `end`, its own node kept; what lies beyond stays
   in [from] under a fresh node. A block carved into objects front to back. */
static inline void sublet_carve(sublet_cap *from, unsigned long end, sublet_cap *to) {
  sublet_cap rest;
  if (end < sublet_end(from)) {
    sublet_split(from, end, &rest);
    sublet_move(from, to);
    sublet_move(&rest, from);
  } else {
    sublet_move(from, to);
  }
}

/* [slot] emptied: what it held is dead or has moved */
static inline void sublet_clear(sublet_cap *slot) {
  __asm__ volatile(".insn s 0x5b, 0x4, x0, 0(%0)\n" : : "r"(slot) : "memory");
}

#endif
