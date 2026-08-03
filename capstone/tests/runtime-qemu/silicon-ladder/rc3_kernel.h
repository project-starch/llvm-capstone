#ifndef RC3_H
#define RC3_H
/* interp threshold: 3 plain globals. */
static char q0[2] = { 1, 0 };
static char q1[2] = { 1, 0 };
static char q2[2] = { 1, 0 };
static unsigned rc3_compute(void)
{
  unsigned r = 0;
  r += (unsigned)(unsigned char)q0[0];
  r += (unsigned)(unsigned char)q1[0];
  r += (unsigned)(unsigned char)q2[0];
  return r;
}
#endif
