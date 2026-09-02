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
 * res[1] and res[2] are written only when S12SHAPE_RES0_ONLY is undefined. The QEMU-side app
 * defines it: the ladder convention gives QEMU a 4-byte region, and writing res[1] there is a
 * capability OOB (cause 7) that halts the domain before it reaches the loop. The board app leaves
 * it undefined and gets the progress counter and the final value.
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
 * FIXED 2026-09-03. The first version passed `(void *)res` as the capability base and faulted in
 * the domain prologue with cause 7 before reaching the loop: in the ladder convention `res` is a
 * PLAIN POINTER into the shared payload, not a capability. The working pattern, taken from
 * trapctl_kernel.h, is a 16-byte-aligned STATIC buffer -- in a gp-captable domain a pointer to a
 * global is materialised through the cap table and IS a capability, whereas an argument pointer is
 * not. It must be static and not a local: a 16-byte-aligned local forces dynamic stack realignment
 * this backend cannot legalize.
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

#define S12_CINC80(out, in) __asm__ volatile(".insn i 0x5b, 0x2, %0, 0x80(%1)" : "=r"(out) : "r"(in))
#define S12_STC(base, val)  __asm__ volatile(".insn s 0x5b, 0x4, %1, 0(%0)" :: "r"(base), "r"(val))

#ifndef S12SHAPE_REPS
#define S12SHAPE_REPS 4096
#endif

/* PRESSURE KNOBS. The baseline rung ran 12,288 executions of the shape in a domain on silicon with
   zero faults, so shape + registers + domain is not sufficient. The mechanism needs the STC to be
   STALLED on a full store buffer -- a four-instruction loop plausibly never fills one, however
   often it runs -- and SQLite's window sits in a 4600-instruction function with entirely different
   surrounding traffic. These knobs add that traffic one variable at a time.

   S12SHAPE_BURST   scalar stores to distinct conflicting lines, issued immediately before the
                    shape so the buffer is still draining when the STC arrives. 4096-byte stride
                    puts every one on the same L1D set (32 KiB, 8-way, 16-byte lines), so each must
                    go to memory.
   S12SHAPE_CAPBURST  the same, but capability stores -- an STC occupies the DYN unit, which a
                    scalar store does not, and the DYN unit serialises one op in flight.
   The burst writes at +4096 and beyond, clear of the load slot at +0 and the store slot at +128,
   so a scalar store can never clobber the capability the LDC reads. That confound looked exactly
   like a 100% hit rate when a bare-metal version of this test tripped over it. */
#ifndef S12SHAPE_BURST
#define S12SHAPE_BURST 0
#endif
#ifndef S12SHAPE_CAPBURST
#define S12SHAPE_CAPBURST 0
#endif

/* 16-byte aligned static, not a local, and not the `res` argument -- see the note above. */
__attribute__((aligned(16))) static unsigned char s12shape_buf[256];
#if S12SHAPE_BURST || S12SHAPE_CAPBURST
/* Separate buffer for the pressure traffic, so it cannot touch the two slots above. */
__attribute__((aligned(16))) static unsigned char s12shape_press[65536];
#endif

static void s12shape_run(volatile unsigned long *res)
{
  /* POINTER TYPES, NOT unsigned long. Casting the buffer to an integer STRIPS THE TAG: the
     first attempt used `unsigned long` and QEMU asserted "cincoffsetimm with an UNTAGGED rs1",
     with the compiler having warned about the cast. A capability must stay in a pointer type all
     the way to the asm operand for the backend to materialise it through the cap table. */
  void *slot_cap, *store_slot, *scratch;
  unsigned long i;

  res[0] = 0x5120;

  /* Two distinct slots in the same static buffer: one the LDC reads from at +0, one the STC
     writes to at +128 -- far enough apart that a store can never clobber the capability the load
     reads, which would be the S-06 confound rather than this mechanism. */
  slot_cap = (void *)s12shape_buf;
  S12_CINC80(store_slot, slot_cap);

  /* Seed the load slot with a real capability: store the region cap into itself. */
  S12_STC(slot_cap, slot_cap);

  res[0] = 0x5121;
#ifndef S12SHAPE_RES0_ONLY
  res[1] = 0;
#endif

#if S12SHAPE_BURST || S12SHAPE_CAPBURST
  {
    void *pb = (void *)s12shape_press;
    (void)pb;
  }
#endif

  for (i = 0; i < S12SHAPE_REPS; i++) {
#if S12SHAPE_BURST
    {
      volatile unsigned long *p = (volatile unsigned long *)s12shape_press;
      int b;
      /* WRAP AT 16. p[] is unsigned long, so index b*512 is byte offset b*4096, and the buffer
         is 65536 -- a burst of 32 would reach 126,976 and run off the end. QEMU caught that as a
         capability OOB before it cost a boot. Sixteen distinct lines all land on the same L1D set
         (8-way), so they still all miss; reusing them costs nothing for store pressure. */
      for (b = 0; b < S12SHAPE_BURST; b++) p[((unsigned)b & 15u) * 512] = (unsigned long)b;
    }
#endif
#if S12SHAPE_CAPBURST
    {
      void *cp = (void *)s12shape_press;
      int b;
      for (b = 0; b < S12SHAPE_CAPBURST; b++) { S12_STC(cp, slot_cap); S12_CINC80(cp, cp); }
    }
#endif
    /* THE SHAPE, in the order the SQLite window has it:
         movc a4, zero      -> the null the STC will carry as its forwarded result
         stc  a4, 0(a5)     -> decoder makes a4 this store's scoreboard rd
         ldc  a4, 0(a0)     -> the load that also claims a4
         cincoffsetimm a4   -> the consumer that reads a4 and faults if mis-forwarded          */
    S12_SHAPE(scratch, store_slot, slot_cap);
#ifndef S12SHAPE_RES0_ONLY
    res[1] = i + 1;          /* progress counter -- board only; the QEMU region is 4 bytes */
#endif
  }

#ifndef S12SHAPE_RES0_ONLY
  res[2] = (unsigned long)scratch;
#endif
  res[0] = 0x5122;
}

#endif
