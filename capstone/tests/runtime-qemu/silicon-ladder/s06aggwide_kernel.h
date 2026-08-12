#ifndef S06AGGWIDE_KERNEL_H
#define S06AGGWIDE_KERNEL_H
/* Wider coverage for the guarded aggregate copy: capabilities in NON-ZERO granules, a
 * MULTI-GRANULE copy, and a STACK destination.
 *
 * WHY. s06aggcap proves one capability in granule 0 of a 32-byte copy between two STATICS
 * survives the guard. That leaves three untested axes, and the open SQLite failure is precisely
 * an untagged capability appearing somewhere the existing rungs do not look:
 *
 *   granule INDEX     -- the pass GEPs per granule; an offset bug would spare granule 0 and
 *                        corrupt every later one, which is exactly the shape that passes
 *                        s06aggcap and still breaks a real program
 *   granule COUNT     -- 4 granules, so the block-splitting done per granule is exercised more
 *                        than once and a continuation-block bug has somewhere to show
 *   destination CLASS -- a STACK object rather than a static. The DAG version of this workaround
 *                        had a recorded bug where, for a copy into a stack slot, the plain
 *                        pre-writes vanished entirely and the output was byte-identical to the
 *                        unfixed build. Statics never exposed it.
 *
 * LAYOUT. A `void *` is SIXTEEN bytes here, so the fields do not land where their names suggest
 * and the struct is 96 bytes / 6 granules, not 64 / 4. Actual byte offsets:
 *
 *   0   g0lo    granule 0 LOW      0x4000 (the S-06 trigger)
 *   8   g0hi    granule 0 HIGH     sentinel A
 *   16  c1      granule 1          a genuine capability (whole granule)
 *   32  c1pad   granule 2 LOW
 *   40  g2lo    granule 2 HIGH  <- named "lo", is actually a HIGH half
 *   48  g2hi    granule 3 LOW   <- named "hi", is actually a LOW half
 *   64  c3      granule 4          a genuine capability (whole granule)
 *   80  c3pad   granule 5 LOW
 *
 * The names are kept because the board results are recorded against them, but READ THE OFFSETS,
 * not the names. The unguarded arm returning 237 is exactly right under this layout: the two
 * lost bits are g0hi (offset 8) and g2lo (offset 40), and both of those are HIGH halves of their
 * granules -- which is precisely what a bare ldc/stc loses. A reading based on the names would
 * have called that result inexplicable.
 *
 * retval bitmask, 255 is a fully correct copy. One bit per checked property so a partial result
 * names exactly what broke rather than collapsing to a single number:
 *
 *   bit 0  g0 low  intact        bit 1  g0 high intact
 *   bit 2  g1 still a capability  bit 3  g1 dereferences to the expected value
 *   bit 4  g2 low  intact        bit 5  g2 high intact
 *   bit 6  g3 still a capability  bit 7  g3 dereferences to the expected value
 *
 *   255  correct
 *   Any zero bit in 2/3 or 6/7 means the guard produced an UNTAGGED capability -- the failure
 *   the SQLite fault looks like. Bits 2 and 6 clear together would point at all capabilities;
 *   bit 6 clear with bit 2 set would point at the per-granule offset arithmetic specifically.
 *
 * Capabilities are checked by DEREFERENCING, not merely by inspecting the tag: the LCC type
 * query is the mechanism under test, so it cannot also be the sole judge. The type query is
 * total, so a lost tag is reported rather than faulting, and the run still returns a number
 * naming every other property.
 */

typedef struct {
  unsigned long g0lo, g0hi;
  void *c1;
  unsigned long c1pad;
  unsigned long g2lo, g2hi;
  void *c3;
  unsigned long c3pad;
} s06aggwide_t;

__attribute__((aligned(16))) static unsigned long s06aggwide_t1[2];
__attribute__((aligned(16))) static unsigned long s06aggwide_t3[2];
__attribute__((aligned(16))) static s06aggwide_t s06aggwide_src;

#define S06AGGWIDE_HA 0xAAAA1111AAAA1111UL
#define S06AGGWIDE_HB 0xBBBB2222BBBB2222UL
#define S06AGGWIDE_V1 0x1111111111111111UL
#define S06AGGWIDE_V3 0x3333333333333333UL

#define S06AGGWIDE_LCC_TYPE(out, cap) \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(out) : "r"(cap))

static unsigned s06aggwide_compute(void)
{
  unsigned r = 0;
  unsigned long ty1 = 0, ty3 = 0;

  s06aggwide_t1[0] = S06AGGWIDE_V1;
  s06aggwide_t3[0] = S06AGGWIDE_V3;

  s06aggwide_src.g0lo = 0x4000UL;
  s06aggwide_src.g0hi = S06AGGWIDE_HA;
  s06aggwide_src.c1   = (void *)s06aggwide_t1;
  s06aggwide_src.c1pad = 0UL;
  s06aggwide_src.g2lo = 0x12345678UL;
  s06aggwide_src.g2hi = S06AGGWIDE_HB;
  s06aggwide_src.c3   = (void *)s06aggwide_t3;
  s06aggwide_src.c3pad = 0UL;

  /* THE DESTINATION IS A STACK LOCAL -- the axis statics cannot test. Not marked aligned(16)
     explicitly: an explicitly 16-byte-aligned LOCAL forces dynamic stack realignment, which this
     backend cannot legalize. The struct contains pointers, so it gets capability alignment from
     its own type, which is what a real program relies on too. */
  s06aggwide_t dst;

  dst = s06aggwide_src;          /* the subject: a 4-granule aggregate copy into a stack object */

  if (dst.g0lo == 0x4000UL)     r |= 1u;
  if (dst.g0hi == S06AGGWIDE_HA) r |= 2u;

  S06AGGWIDE_LCC_TYPE(ty1, dst.c1);
  if (ty1 != 7UL) {
    r |= 4u;
    if (((unsigned long *)dst.c1)[0] == S06AGGWIDE_V1) r |= 8u;
  }

  if (dst.g2lo == 0x12345678UL)  r |= 16u;
  if (dst.g2hi == S06AGGWIDE_HB) r |= 32u;

  S06AGGWIDE_LCC_TYPE(ty3, dst.c3);
  if (ty3 != 7UL) {
    r |= 64u;
    if (((unsigned long *)dst.c3)[0] == S06AGGWIDE_V3) r |= 128u;
  }

  return r;
}
#endif /* S06AGGWIDE_KERNEL_H */
