#ifndef S06BNDS_KERNEL_H
#define S06BNDS_KERNEL_H
/* Does a capability spilled to the stack come back from `ldc` with THE SAME BOUNDS?
 *
 * WHY THIS EXISTS WHEN s06spill ALREADY PASSED. s06spill asks whether the reloaded value is still
 * a capability -- it queries the TYPE and nothing else, and returns 0xFFFF meaning all sixteen
 * survived. That answer is real but it is about a different failure. It would return 0xFFFF
 * unchanged if every single capability came back with WRECKED BOUNDS, because a damaged-bounds
 * capability is still a capability and the type query still says so. Its header even records that
 * it deliberately never dereferences the reloaded pointers. So its clean result never tested this.
 *
 * WHAT THE BOARD ACTUALLY SHOWS (2026-08-14, five wedges, four separately compiled binaries, all
 * mcause 29 = CAPABLITY_OUT_OF_BOUND). At vdbeMemClearExternAndSetNull+0x3c:
 *
 *     lhu  a0, 0x24(a0)   ; a SCALAR load off the pointer          -- SUCCEEDS
 *     ldc  a0, 0x0(a0)    ; reload the capability from a stack slot -- SUCCEEDS
 *     ldc  a1, 0x0(a0)    ; dereference it at OFFSET ZERO           -- OUT_OF_BOUNDS
 *
 * mcause 29, not 25: the reloaded value is TAGGED. It is the bounds that are wrong, badly enough
 * that the capability does not contain its own cursor. That is precisely the case s06spill's type
 * query cannot see, and it is why a clean 0xFFFF there did not close the question.
 *
 * METHOD. Sixteen pointer locals, each a genuine capability over the same target, each in its own
 * stack slot at -O0. Base and end are recorded BEFORE the spill from a register-resident copy, a
 * barrier prevents the reload being folded away, and base and end are read again AFTER. A slot
 * passes only if both match.
 *
 * retval = a 16-bit mask, bit i SET if slot i round-tripped with its bounds intact.
 *   0xFFFF (65535)  bounds survive every spill -- this mechanism is exonerated and the SQLite
 *                   fault must be reached some other way
 *   anything else   the zero bits name which slots were damaged, and the rung IS the minimal
 *                   out-of-SQLite reproduction
 *
 * WHAT THIS RUNG CANNOT REACH, measured while trying. The SQLite capability that faults is a FAR
 * CURSOR into a LARGE object -- 0x306b0 into a 256 KiB heap -- whereas every pointer here has
 * cursor == base over a 64-byte target, the easiest case for the compressed format. A variant with
 * a large static target does NOT LINK in this harness at 8 KiB, 64 KiB or 256 KiB: the ladder
 * places globals at a fixed offset and `.text` then overlaps `.capstone_gp_initdesc`. So a clean
 * result here does NOT exonerate cursor-precision loss on large objects; testing that needs a
 * different vehicle than a ladder rung.
 *
 * ALL QUERIES ARE LCC, SO NOTHING IS EVER DEREFERENCED. A damaged capability is reported, not
 * fallen over: the rung returns a number in both directions instead of wedging, which is the only
 * reason it can measure all sixteen slots rather than at most one.
 *
 * The bounds queries are GUARDED BY THE TYPE QUERY inside a single asm block. Selectors 3 and 4
 * raise on a NOT_CAP or sealed operand, so an unguarded read would make the instrument fault on
 * exactly the input it exists to describe -- and branching in C would put a spill between the test
 * and the use, reintroducing the very round trip under test. Same construction as
 * output_capinfo(), for the same reason.
 */

__attribute__((aligned(16))) static unsigned long s06bnds_target[8];

/* type, base and end of one operand, materialised ONCE, with the bounds reads guarded in asm.
   Returns base/end of -1 when the guard declines, which no real bound can collide with. */
#define S06BNDS_Q(t_, b_, e_, cap_)                              \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %3, x1\n\t"      \
                   "li   %1, -1\n\t"                             \
                   "li   %2, -1\n\t"                             \
                   "li   t0, 3\n\t"                              \
                   "bltu t0, %0, 1f\n\t"                         \
                   ".insn r 0x5b, 0x1, 0x4, %1, %3, x3\n\t"      \
                   ".insn r 0x5b, 0x1, 0x4, %2, %3, x4\n\t"      \
                   "1:\n\t"                                      \
                   : "=&r"(t_), "=&r"(b_), "=&r"(e_)             \
                   : "r"(cap_) : "t0")

/* Not a hardware barrier: it stops the reload being folded into the store, which would make the
   rung pass without ever spilling anything -- a gate that cannot fire. */
#define S06BNDS_BARRIER() __asm__ volatile("" ::: "memory")

#define S06BNDS_DECL(i) void *p##i = (void *)s06bnds_target;
#define S06BNDS_CHECK(i)                                         \
  do {                                                           \
    unsigned long t_, b_, e_;                                    \
    S06BNDS_Q(t_, b_, e_, p##i);                                 \
    if (t_ <= 3UL && b_ == b0_ && e_ == e0_) r |= (1u << (i));   \
  } while (0)

static unsigned s06bnds_compute(void)
{
  unsigned r = 0;
  unsigned long t0_, b0_, e0_;

  s06bnds_target[0] = 0x5A5A5A5A5A5A5A5AUL;

  /* The reference, taken from a fresh capability before any of the locals are spilled. If THIS
     one is already damaged the rung would compare wrong against wrong and pass; guarding on
     t0_ <= 3 and on the sentinel keeps that from reading as a clean result. */
  {
    void *ref = (void *)s06bnds_target;
    S06BNDS_Q(t0_, b0_, e0_, ref);
  }
  if (t0_ > 3UL || b0_ == (unsigned long)-1)
    return 0xEEEEu;   /* the reference itself is unqueryable -- report, never pass silently */

#ifdef S06BNDS_SELFTEST
  /* POSITIVE CONTROL. A rung that returns 0xFFFF proves nothing until the comparison is shown to
     be able to return anything else -- and "bounds match" is exactly the kind of check that
     passes vacuously if the operands are wrong (both -1 from a declined guard, both zero, the
     same reload compared with itself). Perturbing the reference by one byte must drive the mask
     to 0. Build with -DS06BNDS_SELFTEST and require 0; any other value means the comparison is
     not measuring what it claims and no clean run of this rung can be believed. */
  b0_ += 1UL;
#endif

  S06BNDS_DECL(0)  S06BNDS_DECL(1)  S06BNDS_DECL(2)  S06BNDS_DECL(3)
  S06BNDS_DECL(4)  S06BNDS_DECL(5)  S06BNDS_DECL(6)  S06BNDS_DECL(7)
  S06BNDS_DECL(8)  S06BNDS_DECL(9)  S06BNDS_DECL(10) S06BNDS_DECL(11)
  S06BNDS_DECL(12) S06BNDS_DECL(13) S06BNDS_DECL(14) S06BNDS_DECL(15)

  S06BNDS_BARRIER();

  S06BNDS_CHECK(0);  S06BNDS_CHECK(1);  S06BNDS_CHECK(2);  S06BNDS_CHECK(3);
  S06BNDS_CHECK(4);  S06BNDS_CHECK(5);  S06BNDS_CHECK(6);  S06BNDS_CHECK(7);
  S06BNDS_CHECK(8);  S06BNDS_CHECK(9);  S06BNDS_CHECK(10); S06BNDS_CHECK(11);
  S06BNDS_CHECK(12); S06BNDS_CHECK(13); S06BNDS_CHECK(14); S06BNDS_CHECK(15);

  return r;
}
#endif /* S06BNDS_KERNEL_H */
