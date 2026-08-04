#ifndef OB3_H
#define OB3_H
/* Self-contained bounds test: measures the capability's ACTUAL bounds and performs the
 * out-of-bounds store IN THE SAME DOMAIN, on the same capability.
 * The earlier pair (pk2 = 16, oob = 42) split these across two different images, so the "16
 * bytes" and the "did not trap" were never established for the SAME capability -- a real gap
 * once the claim is "the hardware does not enforce".
 * Sequence: load slot 0, read its start/end with lcc, compute headroom, THEN store 8 bytes at
 * +16 and return an encoding of both facts.
 * Return: headroom_bytes * 100 + 42   e.g. 1642 = 16-byte capability AND the store completed.
 *   ...42 with headroom 16  -> a genuine out-of-bounds store that did not trap
 *   ...42 with headroom >16 -> NOT out of bounds; the capability is bigger than assumed and the
 *                              whole finding is void
 *   no result               -> the store trapped; enforcement works */
#define LCC(rd, rs, f) __asm__ volatile(".insn r 0x5b,0x1,0x4, %0, %1, x" #f : "=r"(rd) : "r"(rs))
static char ob3_g[2] = { 1, 0 };
static unsigned ob3_compute(void)
{
  void *c0;
  unsigned long s = 0, e = 0, cur = 0, head;
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(gp)" : "=r"(c0));
  LCC(s, c0, 3); LCC(e, c0, 4); LCC(cur, c0, 2);
  head = e - cur;                       /* bytes available from the cursor to the end */
  if (head > 9000u) head = 9000u;
  __asm__ volatile("sd x0, 16(%0)" :: "r"(c0) : "memory");   /* 8 bytes at +16 */
  (void)ob3_g; (void)s;
  return (unsigned)(head * 100u + 42u);
}
#endif
