#ifndef WBARE_H
#define WBARE_H
/* DECIDES compiler-vs-silicon for the cap-init NULL. wcap used `wcap_data + 16` for its second
 * pointer, and -capstone-cap-init-print shows that leaf emitted with an EMPTY value -- a
 * compiler bug (base+offset initialisers lose their value). Here BOTH initialisers are bare
 * symbols, so both leaves must resolve.
 *   still NULL on board -> an independent SILICON fault, two-line repro
 *   returns 4           -> the compiler bug was the whole story
 * Expect 4. */
static char wbare_data[64] = { 'A', 0 };
static char *wbare_p1 = wbare_data;
static char *wbare_p2 = wbare_data;
static unsigned wbare_compute(void)
{
  unsigned n = 0;
  if (wbare_p1) n += (unsigned)(unsigned char)wbare_p1[0];   /* 'A' = 65 */
  if (wbare_p2) n += 1u;
  return n - 62u;                                            /* 65 + 1 - 62 = 4 */
}
#endif
