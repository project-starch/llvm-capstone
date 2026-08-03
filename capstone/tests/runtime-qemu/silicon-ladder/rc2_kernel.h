#ifndef RC2_H
#define RC2_H
/* interp threshold: 2 plain globals. */
static char q0[2] = { 1, 0 };
static char q1[2] = { 1, 0 };
static unsigned rc2_compute(void)
{
  unsigned r = 0;
  r += (unsigned)(unsigned char)q0[0];
  r += (unsigned)(unsigned char)q1[0];
  return r;
}
#endif
