#ifndef OB7_H
#define OB7_H
/* CONTROL: same shape, but the store is IN BOUNDS (+0). Must return.
 * Sentinel 47 distinguishes this arm from ob3's 42. Return = headroom*100 + 47. */
#define LCC(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static char ob7_g[2] = { 1, 0 };
static unsigned ob7_compute(void)
{
  void *c0;
  unsigned long e = 0, cur = 0, head;
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(gp)" : "=r"(c0));
  LCC(e, c0, 4); LCC(cur, c0, 2);
  head = e - cur;
  if (head > 9000u) head = 9000u;
  /* 32 nops force the producer of c0 to retire and write back, so the store's operand
   * is read from the REGISTER FILE, not the capability-metadata forwarding network. */
  __asm__ volatile(
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "nop\n"
  "sd x0, 0(%0)\n"
  :: "r"(c0) : "memory");
  (void)ob7_g;
  return (unsigned)(head * 100u + 47u);
}
#endif
