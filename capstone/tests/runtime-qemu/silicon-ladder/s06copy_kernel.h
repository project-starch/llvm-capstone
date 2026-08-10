#ifndef S06COPY_H
#define S06COPY_H
/* S-06 standalone reproducer: does a 16-byte ldc/stc block copy preserve all 128 bits of
 * PLAIN UNTAGGED data?
 *
 * This is the aligned middle loop of memcpy, reduced to the smallest thing that can run as a
 * ladder rung. That loop exists so copying a struct containing pointers preserves capability
 * TAGS; it is only correct if the same round trip is bit-exact for ordinary data.
 *
 * RETURN VALUE IS THE VERDICT -- it counts how many of 32 copied bytes came back correct:
 *
 *   32  every byte survived              -> no defect
 *   16  exactly half survived            -> S-06: each 16-byte chunk keeps its LOW 8 bytes
 *                                           and loses its HIGH 8
 *
 * The two counts are far apart and neither is a value the rung could return by accident, so
 * this cannot be misread the way a 0/1 flag can. QEMU returns 32 (it carries an explicit
 * scalar_hi shadow field for this case); the board returns 16.
 *
 * Both halves of buf are 16-byte aligned because buf itself is, so the copy takes the aligned
 * path and neither pointer needs a head loop -- the same condition under which the defect was
 * measured in SQLite.
 *
 * The volatile pointer is load-bearing, not decoration: without it the compiler is entitled to
 * prove the destination equals the source and fold the whole check away, and the rung would
 * then return 32 on defective hardware while testing nothing. The COPY itself deliberately
 * uses plain pointers, so it still lowers to ldc/stc.
 */
/* buf is STATIC, not a local. A 16-byte-aligned LOCAL forces dynamic stack realignment, and
 * on this target that lowers to address arithmetic the backend cannot legalize -- clang dies in
 * LegalizeDAG with "Unable to legalize non-vector shift", at -O0 as well as -O1. A static gets
 * its alignment from the linker for free. */
__attribute__((aligned(16))) static unsigned char s06copy_buf[64];

static unsigned s06copy_compute(void)
{
  unsigned char *buf = s06copy_buf;
  volatile unsigned char *v = buf;
  unsigned i, ok = 0;

  for (i = 0; i < 32u; i++)
    v[i] = (unsigned char)(0xC0u + i);          /* source: 32 bytes of ordinary data */
  for (i = 0; i < 32u; i++)
    v[32u + i] = 0u;                            /* destination, cleared */

  /* THE CONSTRUCT UNDER TEST -- memcpy's aligned middle loop, verbatim. */
  for (i = 0; i + sizeof(void *) <= 32u; i += (unsigned)sizeof(void *))
    *(void **)(buf + 32u + i) = *(void *const *)(buf + i);

  for (i = 0; i < 32u; i++)
    if (v[32u + i] == (unsigned char)(0xC0u + i))
      ok++;
  return ok;
}
#endif
