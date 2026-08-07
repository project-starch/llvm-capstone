#include "fdreg_kernel.h"

/* Dump the stack region fdreg_compute just used, FROM domain_main's frame.
 *
 * Every in-frame instrument perturbed the fault away. A sentinel array (stages 10/11) added 32
 * bytes and the shift-8 build went from 567 to a correct 576; taking `&qc` (stage 12) added a
 * 16-byte capability pointer and did the same. That is not a coincidence to work around -- the
 * frame OFFSET is the variable, so anything that changes the frame changes the experiment.
 *
 * This reads from a DIFFERENT frame. fdreg_compute is called and returns; its frame is dead but
 * its bytes are still there, just below the current stack pointer. domain_main then copies that
 * region into the shared region, which the controller already prints as dbg0.. (res[3..47], see
 * ladder_perf_ctl.c:196-211). fdreg_compute itself is byte-identical to the failing build --
 * verify that by diffing its disassembly against the plain fpga app before trusting a result.
 *
 * res[0] = the compute's return value (so the fault is visible in the same run)
 * res[3..35] = 32 words of the dead frame, low address first
 */
void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned v = fdreg_compute();

  /* A local in THIS frame, taken AFTER the call: its address sits just above the region
     fdreg_compute used, so reading downward from it walks the dead frame. Using an ordinary
     local avoids naming sp in asm, which the capability ABI does not expose as a plain GPR. */
  volatile unsigned here = 0;
  volatile unsigned *w = (volatile unsigned *)&here;

  res[0] = (unsigned long)v;
  res[1] = 0UL;
  res[2] = 0xD09EUL;
  {
    int i;
    /* Downward from this frame into the dead one. */
    for (i = 0; i < 32; i++)
      res[3 + i] = (unsigned long)w[-i];
  }
}
