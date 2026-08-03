#ifndef RC16_H
#define RC16_H
/* interp-glue threshold bisect: 16 plain globals. */
static char q0[2] = { 1, 0 };
static char q1[2] = { 1, 0 };
static char q2[2] = { 1, 0 };
static char q3[2] = { 1, 0 };
static char q4[2] = { 1, 0 };
static char q5[2] = { 1, 0 };
static char q6[2] = { 1, 0 };
static char q7[2] = { 1, 0 };
static char q8[2] = { 1, 0 };
static char q9[2] = { 1, 0 };
static char q10[2] = { 1, 0 };
static char q11[2] = { 1, 0 };
static char q12[2] = { 1, 0 };
static char q13[2] = { 1, 0 };
static char q14[2] = { 1, 0 };
static char q15[2] = { 1, 0 };
static unsigned rc16_compute(void)
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
  r += (unsigned)(unsigned char)q8[0];
  r += (unsigned)(unsigned char)q9[0];
  r += (unsigned)(unsigned char)q10[0];
  r += (unsigned)(unsigned char)q11[0];
  r += (unsigned)(unsigned char)q12[0];
  r += (unsigned)(unsigned char)q13[0];
  r += (unsigned)(unsigned char)q14[0];
  r += (unsigned)(unsigned char)q15[0];
  return r;
}
#endif
