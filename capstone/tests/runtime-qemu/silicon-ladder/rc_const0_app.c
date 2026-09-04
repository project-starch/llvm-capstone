/* Reconstruction of the July rc_const0 / rc_p1 pair (the originals are not in
 * the record): a loop that stores to a GLOBAL array and keeps a live accumulator
 * in the same body.  rc_const0 stores the loop index; rc_p1 stores index+1.
 * Ladder convention: result in res[], through a volatile pointer so nothing is
 * sunk or reordered. */
static long acc[64];
void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  volatile unsigned long *out = res;
  long n = 64;
  __asm__ volatile("" : "+r"(n));
  long s = 0;
  int i;
  for (i = 0; i < n; i++) { acc[i] = i; s += acc[i]; }
  out[0] = (unsigned long)s;
}
