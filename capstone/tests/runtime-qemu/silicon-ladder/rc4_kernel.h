#ifndef RC4_H
#define RC4_H
/* interp-glue threshold bisect: 4 plain globals. */
static char q0[2] = { 1, 0 };
static char q1[2] = { 1, 0 };
static char q2[2] = { 1, 0 };
static char q3[2] = { 1, 0 };
static unsigned rc4_compute(void)
{
  unsigned r = 0;
  r += (unsigned)(unsigned char)q0[0];
  r += (unsigned)(unsigned char)q1[0];
  r += (unsigned)(unsigned char)q2[0];
  r += (unsigned)(unsigned char)q3[0];
  return r;
}
#endif
