#ifndef RC8_H
#define RC8_H
/* interp-glue threshold bisect: 8 plain globals. */
static char q0[2] = { 1, 0 };
static char q1[2] = { 1, 0 };
static char q2[2] = { 1, 0 };
static char q3[2] = { 1, 0 };
static char q4[2] = { 1, 0 };
static char q5[2] = { 1, 0 };
static char q6[2] = { 1, 0 };
static char q7[2] = { 1, 0 };
static unsigned rc8_compute(void)
{
  unsigned r = 0;
  r += (unsigned)(unsigned char)q0[0];
  r += (unsigned)(unsigned char)q1[0];
  r += (unsigned)(unsigned char)q2[0];
  r += (unsigned)(unsigned char)q3[0];
  r += (unsigned)(unsigned char)q4[0];
  r += (unsigned)(unsigned char)q5[0];
  r += (unsigned)(unsigned char)q6[0];
  r += (unsigned)(unsigned char)q7[0];
  return r;
}
#endif
