#ifndef K1200_H
#define K1200_H
/* Boundary pair for the R-14 conjunction. IDENTICAL source except the padding size, which is
 * dead. Offline scan of the emitted code: pad=800 -> frame 3776 with ZERO lui-based frame
 * addressing sites; pad=1200 -> frame 4576 with 13 of them. So this pair isolates the exact
 * compiler-observable predicate 'does the frame need lui to be addressed', which is NOT the
 * same as raw frame size (f2nop at frame 2176 has zero lui sites and PASSES). Expect 4.
 * k1200: the lui-USING side. */
static unsigned k1200_len(const char *s){unsigned n=0;while(s&&s[n])n++;return n;}
static unsigned k1200_compute(void)
{
  struct kv { const char *z; const char *y; };
  struct kv a[32];
  volatile char pad[1200];
  unsigned i; int ok = 0;
  pad[0] = 1;
  for (i = 0; i < 4; i++) { a[i].z = "x0"; a[i].y = "y0"; }
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && k1200_len(a[i].z) > 0 && k1200_len(a[i].y) > 0) ok++;
  return (unsigned)ok + (unsigned)pad[0] - 1u;
}
#endif
