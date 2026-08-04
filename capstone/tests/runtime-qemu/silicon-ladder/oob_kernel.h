#ifndef OOB_H
#define OOB_H
/* DIRECT test of capability upper-bound enforcement on stores.
 * pk2 = 16 established that cap-table carves are ADJACENT 16-byte regions (end1 == start0,
 * end0-start0 == 16), so the 6-byte overrun observed during the pad investigation was a genuine
 * out-of-bounds write into the neighbouring carve -- and it did not fault.
 * load_store_unit.sv:970-972 implements an exact upper-bound check
 * (`lsu_ea_full + lsu_access_sz > bound_end` -> cause 28), so this SHOULD trap.
 * Here we do it deliberately: take the slot-0 storage capability (16 bytes) and store 8 bytes at
 * offset 16 -- one byte past the end, entirely outside the bounds.
 *   returns 42 -> the store did NOT trap: the upper bound is NOT enforced on this path.
 *                 That is a capability-enforcement hole, not a software bug.
 *   wedges     -> it DID trap; the earlier overrun must have been in bounds after all and the
 *                 pk2 reading needs revisiting.
 * Deliberately the LAST rung in its boot: it is the one expected to fault. */
static char oob_g[2] = { 1, 0 };
static unsigned oob_compute(void)
{
  void *c0;
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(gp)" : "=r"(c0));   /* ldc gp[0] : 16-byte carve */
  /* sd zero at +16 -- one past the capability's end */
  __asm__ volatile("sd x0, 16(%0)" :: "r"(c0) : "memory");
  (void)oob_g;
  return 42u;
}
#endif
