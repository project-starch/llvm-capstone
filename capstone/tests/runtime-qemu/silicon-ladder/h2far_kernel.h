#ifndef H2FAR_KERNEL_H
#define H2FAR_KERNEL_H
/* Refining the R-14 conjunction (big lui-addressed frame AND a loop storing TWO capabilities
 * per iteration to a computed element). These drop the STRUCT entirely -- a flat pointer
 * array -- so if they still fail, the struct is irrelevant and the repro is smaller again.
 * h2far: same two stores per iteration but to FAR-APART slots p[i], p[i+8]
 * (offsets 0x0 and 0x80 rather than 0x0 and 0x10). Isolates the OFFSET PAIR. Expect 524. */
static char g0[2] = { 'A', 0 };
static char g1[2] = { 'B', 0 };
static unsigned h2far_compute(void)
{
  const char *p[16];
  volatile char pad[2200];
  unsigned i, r = 0;
  pad[0] = 1;
  for (i = 0; i < 4; i++) { p[i] = g0; p[i+8] = g1; }
  for (i = 0; i < 4; i++) r += (unsigned)(unsigned char)p[i][0] + (unsigned)(unsigned char)p[i+8][0];
  return r + (unsigned)pad[0] - 1u;
}
#endif
