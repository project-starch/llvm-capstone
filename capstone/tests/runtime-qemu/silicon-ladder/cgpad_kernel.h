#ifndef CGPAD_KERNEL_H
#define CGPAD_KERNEL_H
/* cgs8 (which PASSES) plus dead padding to push the frame past 2048.
 * Valid results so far: small frame => always passes; big frame is NECESSARY but NOT
 * sufficient (r14sl frame 4256 and r14hl frame 4288 both pass). Every valid failure has a big
 * frame AND an `ldc` from the cap-table inside the capability-store loop. cgs8 has that loop
 * and that load but a SMALL frame, and passes. Adding padding gives it the big frame too:
 * if cgpad FAILS, the trigger is the CONJUNCTION (big frame + ldc-from-gp in the store loop),
 * and neither alone. Expect 520. */
static char g0[2] = { 'A', 0 };
static unsigned cgpad_compute(void)
{
  const char *p[8];
  volatile char pad[2200];
  unsigned i, r = 0;
  pad[0] = 1;
  for (i = 0; i < 8; i++) p[i] = g0;
  for (i = 0; i < 8; i++) r += (unsigned)(unsigned char)p[i][0];
  return r + (unsigned)pad[0] - 1u;
}
#endif
