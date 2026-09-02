#ifndef S12SHAPE_KERNEL_H
#define S12SHAPE_KERNEL_H
/* S-12: does the faulting SHAPE fault INSIDE A REAL CAPABILITY DOMAIN on silicon?
 *
 * WHY THIS RUNG EXISTS. The board has localised S-12 to one instruction operand: the fault needs
 * the capability store's source, the following load's destination, and the faulting instruction's
 * operand to be the SAME architectural register. An RTL mechanism predicts exactly that --
 * decoder.sv:1313 decodes STC's rd := rs2, so the store is a real scoreboard PRODUCER whose value
 * for a null capability is create_cnull() = {cursor 0, NOT_CAP}, i.e. mcause 25 with tval 0; a
 * store-buffer-stalled STC never retracts we_gpr (commit_stage.sv:331-347) so it stays a live
 * claimant; and forwarding candidacy needs still_issued & sbe.valid, so a written-back STC beats
 * an unproduced LDC.
 *
 * The mechanism does NOT fire in bare-metal Verilator. Three variants -- warm, cache-missing, and
 * store-buffer-pressured -- all report the tree's own precondition counter firing every iteration
 * (S12-ESC escape=1026/258, which is its own positive control) with the outcome counter at 0. So
 * the instruction shape alone is not sufficient, and the untested variable is the one Verilator
 * cannot supply: a real capability domain -- capenter, a monitor-carved stack, globals reached
 * through a cap table, and the monitor's own store traffic.
 *
 * This rung supplies exactly that and nothing else. It is ~10 KB against SQLite's 1.6 MB, so a
 * fault here is a minimal reproducer; a clean run says the domain context is not the missing
 * ingredient either, and sends the search to what SQLite specifically adds.
 *
 * MUST be built with INTERP_DOMAIN_MTVEC=1. Without the in-domain trap vector a fault wedges the
 * core and "no return" cannot be told from "no fault" -- the exact ambiguity that made every
 * earlier reconstruction unreadable.
 *
 * VERDICT, read from res[0], each written BEFORE the step it labels:
 *   0x5120  entered, nothing else reached -- wedged before the shape ran.
 *   0x5121  the capability was created and stored; the loop is about to run.
 *   0x5122  THE LOOP COMPLETED WITHOUT FAULTING. The shape does not fault in a domain either,
 *           at this iteration count. res[1] carries the count actually executed.
 *   a word of the form 0xF...  the glue's trap handler fired: bits 27..22 are mcause and bits
 *           21..0 are (mepc - _start) >> 2. mcause 25 with the mepc landing on the cincoffsetimm
 *           IS S-12 REPRODUCED IN A 10 KB DOMAIN.
 *   NO RETURN AT ALL  the handler did not fire; the run is VOID, not a negative. Check trapctl.
 *
 * STATUS 2026-09-03: NOT YET RUNNING. The QEMU verify step faults before reaching the loop --
 * `cause = 7` (capability OOB) on the domain prologue's own `stc ra, 0x30(sp)` at VA 0x10254. The
 * cause is this file, not the hardware: `region` is passed as `(void *)res`, but in the ladder
 * convention `res` is a PLAIN POINTER into the shared payload, not a capability, so using it as an
 * STC/LDC base is wrong and the frame/stack assumptions do not hold. The fix is to obtain a proper
 * region capability the way the other rungs do rather than casting `res`. Everything above this
 * line is analysis and still stands; the rung itself must not be boarded until it runs under QEMU.
 *
 * The value stored is a NULL capability, deliberately: that is what makes the STC's forwarded
 * result {cursor 0, NOT_CAP} under the mechanism, and it is what the SQLite window does at [32].
 */

/* THE FOUR INSTRUCTIONS MUST BE ONE ASM BLOCK WITH ONE TIED REGISTER.
   Written as four separate __asm__ statements the compiler spills between them and allocates
   different registers to each -- the built binary had `ldc a1` followed by a stack round-trip and
   `cincoffsetimm a0, a0`, so the store source, the load destination and the consumer's operand
   were three DIFFERENT registers and the rung tested nothing. The whole localisation says the
   fault needs them to be the SAME register, so %0 is used for all four and they sit in one block
   with no compiler-inserted code between them. "=&r" makes %0 an early-clobber so it cannot be
   allocated on top of %1 or %2. */
#define S12_SHAPE(scratch, store_slot, load_slot)                                  \
  __asm__ volatile(                                                                \
      ".insn r 0x5b, 0x1, 0xa, %0, x0, x0\n\t"   /* movc          %0, zero      */ \
      ".insn s 0x5b, 0x4, %0, 0(%1)\n\t"         /* stc           %0, 0(store)  */ \
      ".insn i 0x5b, 0x3, %0, 0(%2)\n\t"         /* ldc           %0, 0(load)   */ \
      ".insn i 0x5b, 0x2, %0, 0xb0(%0)"          /* cincoffsetimm %0, %0, 0xb0  */ \
      : "=&r"(scratch) : "r"(store_slot), "r"(load_slot))

#define S12_CINCI(out, in)  __asm__ volatile(".insn i 0x5b, 0x2, %0, 0xb0(%1)" : "=r"(out) : "r"(in))
#define S12_STC(base, val)  __asm__ volatile(".insn s 0x5b, 0x4, %1, 0(%0)" :: "r"(base), "r"(val))

#ifndef S12SHAPE_REPS
#define S12SHAPE_REPS 4096
#endif

static inline void s12shape_run(volatile unsigned long *res, void *region)
{
  unsigned long slot_cap, store_slot, scratch;
  unsigned long i;

  res[0] = 0x5120;

  /* region is the shared payload capability the glue hands us. Two distinct slots inside it:
     one the LDC reads from, one the STC writes to -- kept apart so a scalar clobber of the
     loaded capability can never be mistaken for the mechanism (the S-06 confound). */
  slot_cap   = (unsigned long)region;
  S12_CINCI(store_slot, slot_cap);      /* store target, +0xb0 clear of the load slot */

  /* Seed the load slot with a real capability: store the region cap into itself. */
  S12_STC(slot_cap, slot_cap);

  res[0] = 0x5121;
  res[1] = 0;

  for (i = 0; i < S12SHAPE_REPS; i++) {
    /* THE SHAPE, in the order the SQLite window has it:
         movc a4, zero      -> the null the STC will carry as its forwarded result
         stc  a4, 0(a5)     -> decoder makes a4 this store's scoreboard rd
         ldc  a4, 0(a0)     -> the load that also claims a4
         cincoffsetimm a4   -> the consumer that reads a4 and faults if mis-forwarded          */
    S12_SHAPE(scratch, store_slot, slot_cap);
    res[1] = i + 1;
  }

  res[2] = scratch;
  res[0] = 0x5122;
}

#endif
