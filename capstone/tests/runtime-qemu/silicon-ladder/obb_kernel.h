#ifndef OBB_H
#define OBB_H
/* Out-of-bounds LOAD at +16. The LSU checks LOAD and STORE in the same block, via
 * perm[2] instead of perm[1]. If loads trap and stores do not, the defect is store-side. */
#define LCC(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static char obb_g[2] = { 1, 0 };
static unsigned obb_compute(void)
{
  void *c0;
  unsigned long e = 0, cur = 0, head;  unsigned long v = 0;
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
  "ld %0, 16(%1)\n"
  : "=r"(v) : "r"(c0) : "memory");
  (void)obb_g;
  return (unsigned)(head * 100u + 53u);
}
#endif
