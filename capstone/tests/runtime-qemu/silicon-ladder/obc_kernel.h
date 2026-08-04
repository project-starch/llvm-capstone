#ifndef OBC_H
#define OBC_H
/* CONTROL: 8-byte store at +8 -> [base+8,base+16), the LAST fully in-bounds slot.
 * Pins down the boundary arithmetic: this MUST succeed if the capability really is 16 B. */
#define LCC(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static char obc_g[2] = { 1, 0 };
static unsigned obc_compute(void)
{
  void *c0;
  unsigned long e = 0, cur = 0, head; 
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(gp)" : "=r"(c0));
  LCC(e, c0, 4); LCC(cur, c0, 2);
  head = e - cur;
  if (head > 9000u) head = 9000u;
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
  "sd x0, 8(%0)\n"
  :: "r"(c0) : "memory");
  (void)obc_g;
  return (unsigned)(head * 100u + 55u);
}
#endif
