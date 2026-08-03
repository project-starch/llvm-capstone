#ifndef H2ADJ_KERNEL_H
#define H2ADJ_KERNEL_H
/* Refining the R-14 conjunction (big lui-addressed frame AND a loop storing TWO capabilities
 * per iteration to a computed element). These drop the STRUCT entirely -- a flat pointer
 * array -- so if they still fail, the struct is irrelevant and the repro is smaller again.
 * h2adj: two stores per iteration to ADJACENT flat slots p[2i], p[2i+1]. Expect 524. */
static char g0[2] = { 'A', 0 };
static char g1[2] = { 'B', 0 };
static unsigned h2adj_compute(void)
{
  const char *p[16];
  volatile char pad[2200];
  unsigned i, r = 0;
  pad[0] = 1;
  for (i = 0; i < 4; i++) { p[2*i] = g0; p[2*i+1] = g1; }
  for (i = 0; i < 8; i++) r += (unsigned)(unsigned char)p[i][0];
  return r + (unsigned)pad[0] - 1u;
}
#endif
