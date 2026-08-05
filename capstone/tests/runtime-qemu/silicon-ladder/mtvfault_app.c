#include "mtvfault_kernel.h"

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  *res = mtvfault_expect();      /* 17, written BEFORE the fault */

  /* Deliberate capability fault: `ldc t1, 0(x0)`. x0 carries no capability, so this is a
     capability violation rather than a plain address fault -- the same class the domain
     hits at its own exit (cause 24), and not the illegal-instruction cause 2 that the
     trap latch excludes.

     The "memory" clobber is load-bearing. Without it the store above is dead (immediately
     overwritten below) and the compiler deletes it, so a working handler would report 2989
     and read as "no fault". A probe whose success case is optimised away is exactly how
     four sessions of "measurements" were produced by a broken instrument. */
  asm volatile(".insn i 0x5b, 0x3, t1, 0(x0)" ::: "t1", "memory");

  *res = 2989u;                  /* 0xBAD -- must NOT be reached */
}
