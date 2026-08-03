#ifndef MG2_H
#define MG2_H
/* Localise WHICH descriptor iteration the interp glue breaks on.
 * Each global holds a DISTINCT value (1,2,3...) and is checked individually, so the
 * return value is a COUNT of correctly-initialised globals -- not a sum that hides which
 * one failed. Measured: two char[2] globals return 0; with the copy length rounded to 8,
 * exactly ONE of the two becomes correct. That asymmetry between identical globals is the
 * open clue. Expect 2. */
static char m0[2] = { 1, 0 };
static char m1[2] = { 2, 0 };
static unsigned mg2_compute(void)
{
  unsigned ok = 0;
  if ((unsigned char)m0[0] == 1u) ok++;
  if ((unsigned char)m1[0] == 2u) ok++;
  return ok;
}
#endif
