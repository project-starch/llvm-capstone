#ifndef OB5_H
#define OB5_H
/* The discriminator: out-of-bounds store at +16 with the producer retired.
 * If this TRAPS while ob3 (back-to-back) did not, the fault is capability-metadata
 * FORWARDING -- already fixed upstream by 7aac52f93 -- not a missing bounds check.
 * Sentinel 45 distinguishes this arm from ob3's 42. Return = headroom*100 + 45. */
#define LCC(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static char ob5_g[2] = { 1, 0 };
static unsigned ob5_compute(void)
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
  "sd x0, 16(%0)\n"
  :: "r"(c0) : "memory");
  (void)ob5_g;
  return (unsigned)(head * 100u + 45u);
}
#endif
