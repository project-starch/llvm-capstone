#ifndef S06LCC_KERNEL_H
#define S06LCC_KERNEL_H
/* SILICON CONFIRMATION for the S-06 enabler: is LCC's TYPE query TOTAL on this bitstream?
 *
 * The RTL change (capstone-ariane fpga-testing-dev-s06, one line in capstone_dyn_unit.anvil)
 * makes `lcc rd, rs, 1` answer 7 for a NON-CAPABILITY instead of raising UNEXPECTED_OPERAND.
 * Verilator says it works. This asks the silicon.
 *
 * IT DOUBLES AS A BITSTREAM IDENTITY CHECK, which is the reason to run it first. On the OLD
 * bitstream the plain-data query RAISES, and a capability fault inside a domain WEDGES rather
 * than trapping -- so an old bitstream produces NO RETURN AT ALL, not a wrong number. The two
 * outcomes cannot be confused:
 *
 *   returns 171  -> the new bitstream is live and the query behaves as designed
 *   no return    -> either the old bitstream is still resident, or the change does not work
 *                   on silicon. Distinguish with the k800 control in the same boot: if k800
 *                   returns and this does not, the board is fine and this is a real result.
 *
 * THE VERDICT IS ENCODED SO EACH DIGIT IS INDEPENDENTLY MEANINGFUL, because a single flag
 * cannot say WHICH half failed:
 *
 *   hundreds : type query on a REAL NONLIN capability   expect 1  (spec numbering, NOT 7)
 *   tens     : type query on PLAIN untagged data        expect 7  (the change under test)
 *   units    : the 16-byte copy came back intact        expect 1
 *
 * So 171 is a full pass. 071 would mean the query answers 7 for everything -- which would also
 * "pass" a naive plain-data-only test, and is exactly why the capability arm is included. 170
 * would mean the query works but the copy still loses data. 1x0 with x != 7 would mean the
 * change is absent while capabilities still answer correctly, i.e. the old bitstream in the
 * case where the fault somehow did not wedge.
 */

/* Static, 16-byte aligned: a 16-byte-aligned LOCAL forces dynamic stack realignment, which this
 * backend cannot legalize (clang dies in LegalizeDAG). A static gets its alignment from the
 * linker for free. Same reasoning as s06copy_kernel.h. */
__attribute__((aligned(16))) static unsigned char s06lcc_src[32];
__attribute__((aligned(16))) static unsigned char s06lcc_dst[32];

/* LCC selector 1 = query type. The selector rides the rs2 ENCODING field, so it must be written
 * as a register NAME whose number is the selector -- `x1`, never a literal 1. */
#define S06_LCC_TYPE(out, cap) \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(out) : "r"(cap))

static unsigned s06lcc_compute(void)
{
  unsigned long ty_cap = 0, ty_plain = 0;
  unsigned units = 0;

  /* --- tens digit: the query on PLAIN data. On the old bitstream this WEDGES. ---
   * Build genuinely untagged data with ordinary stores, then load it capability-grained. The
   * load forces the metadata to zero, so the value really is NOT_CAP -- fabricating one another
   * way would not exercise the same path. */
  volatile unsigned long *s = (volatile unsigned long *)s06lcc_src;
  s[0] = 0x0123456789abcdefUL;
  s[1] = 0xfedcba9876543210UL;
  {
    void *plain;
    __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(%1)" : "=r"(plain) : "r"(s06lcc_src));
    S06_LCC_TYPE(ty_plain, plain);
  }

  /* --- hundreds digit: the query on a REAL capability. Must NOT answer 7. ---
   * Without this arm a query that answered 7 unconditionally would look like a pass. */
  {
    void *cap = (void *)s06lcc_dst;   /* a real, tagged capability */
    S06_LCC_TYPE(ty_cap, cap);
  }

  /* --- units digit: the actual S-06 repair, using the query to choose the copy. ---
   * Read BOTH halves plainly FIRST, so the data survives regardless of what ldc does, then ask,
   * then write the destination exactly ONCE. Poison the destination first so a half-written
   * result cannot masquerade as success. */
  {
    volatile unsigned long *d = (volatile unsigned long *)s06lcc_dst;
    d[0] = 0xdeadbeefcafef00dUL;
    d[1] = 0xdeadbeefcafef00dUL;

    unsigned long lo = s[0], hi = s[1];      /* plain reads, before any ldc */
    void *chunk;
    unsigned long ty;
    __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(%1)" : "=r"(chunk) : "r"(s06lcc_src));
    S06_LCC_TYPE(ty, chunk);
    if (ty == 7) {
      d[0] = lo;                             /* plain path: one write, both halves */
      d[1] = hi;
    } else {
      __asm__ volatile(".insn s 0x5b, 0x4, %0, 0(%1)" :: "r"(chunk), "r"(s06lcc_dst) : "memory");
    }
    if (d[0] == 0x0123456789abcdefUL && d[1] == 0xfedcba9876543210UL)
      units = 1;
  }

  return (unsigned)(ty_cap * 100u + ty_plain * 10u + units);
}
#endif /* S06LCC_KERNEL_H */
