/* R-8 minimal probe -- and a DISCRIMINATOR, not just a minimisation.
 *
 * R-8/R-6: a scalar accumulated across a loop keeps its INITIAL value while the
 * addend is computed correctly and the loop runs its full trip count.
 *
 * "Hardware cannot accumulate" is an extraordinary claim. An ordinary one fits every
 * observation just as well: the accumulator lives in a REGISTER that something
 * clobbers on silicon -- the entry glue, the cscall path, or a trap handler saving
 * fewer registers than our QEMU models. That would present identically: right
 * addend, right trip count, value reverting to its initial state.
 *
 * These probes separate those. Each is the same 5-line loop; only WHERE the
 * accumulator lives differs.
 *
 *   dbg0  baseline accumulate, compiler's register choice     expect 100
 *   dbg1  same, but accumulator pinned to a CALLEE-SAVED reg (s1)
 *   dbg2  same, pinned to a CALLER-SAVED/temp reg (t2)
 *   dbg3  accumulator forced through MEMORY every iteration (volatile)
 *   dbg4  add a CONSTANT (a = a + 1) rather than +=
 *   dbg5  accumulate a value loaded from a volatile global (addend not constant)
 *   dbg6  trip counter, incremented in the same loop as dbg0   expect 100
 *   dbg7  short loop, 3 trips                                  expect 3
 *   dbg8  multiply-accumulate instead of add                   expect 1024
 *
 * READING IT. If dbg3 (memory) is correct while register forms fail -> the value is
 * being lost in a register, i.e. a save/restore bug in OUR glue or the monitor, NOT
 * a hardware adder fault. If a callee-saved form fails and a temp form passes (or
 * vice versa) that names the register class, which points straight at whoever fails
 * to preserve it. If ALL fail including memory, the fault is genuinely in the
 * loop/accumulate mechanism and the hardware claim stands.
 * If dbg7 passes but dbg0 fails, it is trip-count dependent -- suggesting something
 * periodic (a timer/trap) rather than a static codegen error.
 */
static volatile long ap_one = 1;
static volatile long ap_mem;

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  int i;
  long n = 100;
  __asm__ volatile("" : "+r"(n));

  /* dbg0: plain accumulate, compiler picks the register */
  long a0 = 0;
  for (i = 0; i < n; i++) a0 += 1;
  res[3 + 0] = (unsigned long)a0;

  /* dbg1: accumulator pinned to a callee-saved register */
  register long a1 __asm__("s1") = 0;
  for (i = 0; i < n; i++) { a1 += 1; __asm__ volatile("" : "+r"(a1)); }
  res[3 + 1] = (unsigned long)a1;

  /* dbg2: accumulator pinned to a temp (caller-saved) register */
  register long a2 __asm__("t2") = 0;
  for (i = 0; i < n; i++) { a2 += 1; __asm__ volatile("" : "+r"(a2)); }
  res[3 + 2] = (unsigned long)a2;

  /* dbg3: accumulator lives in MEMORY -- volatile forces a load/store each pass */
  ap_mem = 0;
  for (i = 0; i < n; i++) ap_mem = ap_mem + 1;
  res[3 + 3] = (unsigned long)ap_mem;

  /* dbg4: a = a + 1 rather than += (janne's exact form) */
  long a4 = 0;
  for (i = 0; i < n; i++) a4 = a4 + 1;
  res[3 + 4] = (unsigned long)a4;

  /* dbg5: addend loaded from a volatile global, so it is not a constant */
  long a5 = 0;
  for (i = 0; i < n; i++) a5 += ap_one;
  res[3 + 5] = (unsigned long)a5;

  /* dbg6: independent trip counter in the same loop as dbg0's shape */
  long a6 = 0, trips = 0;
  for (i = 0; i < n; i++) { a6 += 1; trips++; }
  res[3 + 6] = (unsigned long)trips;

  /* dbg7: short loop -- 3 trips */
  long a7 = 0;
  for (i = 0; i < 3; i++) a7 += 1;
  res[3 + 7] = (unsigned long)a7;

  /* dbg8: multiply-accumulate, 2^10 */
  long a8 = 1;
  for (i = 0; i < 10; i++) a8 *= 2;
  res[3 + 8] = (unsigned long)a8;

  res[0] = (unsigned)a0;
  res[2] = 0xD09E;
}
