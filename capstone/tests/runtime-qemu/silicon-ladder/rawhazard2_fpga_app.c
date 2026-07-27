/* Loop-free store->load hazard probe (v2).
 *
 * v1 hung on the board. v1 contained loops (a distance sweep and nop loops), and
 * loops are precisely what is under suspicion, so v1 could not distinguish "the
 * hazard" from "whatever hangs the other rungs". This version has NO loops and no
 * data-dependent branches at all: it is a straight line of
 *   store OLD ; store NEW ; <0,2,4,..,12 nops> ; load back
 * with the result of each load written to its own debug slot.
 *
 * Correct hardware: every slot reads 187 (0xBB = NEW).
 * A stale load reads 170 (0xAA = OLD) and localizes the hazard window.
 * If THIS hangs, the fault is not loop-shaped and the insertsort reading needs
 * revisiting. Either outcome is informative, which v1 could not guarantee.
 */
#include "rawhazard_kernel.h"

static unsigned long rh_rd_mcycle(void) {
  unsigned long v; __asm__ volatile("csrr %0, mcycle" : "=r"(v)); return v;
}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = rh_rd_mcycle();
  rh_a[0] = RH_OLD;
  __asm__ volatile("" ::: "memory");
  rh_a[0] = RH_NEW;
  __asm__ volatile("" ::: "memory");
  __asm__ volatile("" ::: "memory");
  res[3 + 0] = (unsigned long)rh_a[0];
  rh_a[0] = RH_OLD;
  __asm__ volatile("" ::: "memory");
  rh_a[0] = RH_NEW;
  __asm__ volatile("" ::: "memory");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("" ::: "memory");
  res[3 + 1] = (unsigned long)rh_a[0];
  rh_a[0] = RH_OLD;
  __asm__ volatile("" ::: "memory");
  rh_a[0] = RH_NEW;
  __asm__ volatile("" ::: "memory");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("" ::: "memory");
  res[3 + 2] = (unsigned long)rh_a[0];
  rh_a[0] = RH_OLD;
  __asm__ volatile("" ::: "memory");
  rh_a[0] = RH_NEW;
  __asm__ volatile("" ::: "memory");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("" ::: "memory");
  res[3 + 3] = (unsigned long)rh_a[0];
  rh_a[0] = RH_OLD;
  __asm__ volatile("" ::: "memory");
  rh_a[0] = RH_NEW;
  __asm__ volatile("" ::: "memory");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("" ::: "memory");
  res[3 + 4] = (unsigned long)rh_a[0];
  rh_a[0] = RH_OLD;
  __asm__ volatile("" ::: "memory");
  rh_a[0] = RH_NEW;
  __asm__ volatile("" ::: "memory");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("" ::: "memory");
  res[3 + 5] = (unsigned long)rh_a[0];
  rh_a[0] = RH_OLD;
  __asm__ volatile("" ::: "memory");
  rh_a[0] = RH_NEW;
  __asm__ volatile("" ::: "memory");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("nop");
  __asm__ volatile("" ::: "memory");
  res[3 + 6] = (unsigned long)rh_a[0];
  unsigned long c1 = rh_rd_mcycle();
  res[0] = 0xBEEFUL;
  res[1] = c1 - c0;
  res[2] = 0xD09EUL;
}
