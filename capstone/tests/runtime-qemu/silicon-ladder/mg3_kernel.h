#ifndef MG3_H
#define MG3_H
/* Localise WHICH descriptor iteration the interp glue breaks on.
 * Each global holds a DISTINCT value (1,2,3...) and is checked individually, so the
 * return value is a COUNT of correctly-initialised globals -- not a sum that hides which
 * one failed. Measured: two char[2] globals return 0; with the copy length rounded to 8,
 * exactly ONE of the two becomes correct. That asymmetry between identical globals is the
 * open clue. Expect 3. */
static char m0[2] = { 1, 0 };
static char m1[2] = { 2, 0 };
static char m2[2] = { 3, 0 };
static unsigned mg3_compute(void)
{
  unsigned ok = 0;
  if ((unsigned char)m0[0] == 1u) ok++;
  if ((unsigned char)m1[0] == 2u) ok++;
  if ((unsigned char)m2[0] == 3u) ok++;
  return ok;
}
#endif
