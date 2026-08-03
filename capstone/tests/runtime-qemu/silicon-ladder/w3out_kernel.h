#ifndef W3OUT_H
#define W3OUT_H
/* WORKAROUND CANDIDATE for R-14. Baseline: k1200 (struct a[32] + 1200 B dead pad -> frame
 * 4576 B, 13 lui-addressing sites) FAILS on silicon while k800 (800 B pad, frame 3776, 0
 * lui sites) passes. The fault needs a big lui-addressed frame, so a workaround must stop
 * the capability stores happening in one.
 * w3out: the array and the 1200 B pad STAY in the big frame, but the capability stores are
 * outlined into a noinline helper. The helper's own frame is tiny and its base register is a
 * plain ARGUMENT, never a lui-derived frame address. If this passes, the workaround is a
 * source-level outlining of the initialisation -- no data has to move to globals. */
static unsigned w3out_len(const char *s){unsigned n=0;while(s&&s[n])n++;return n;}
struct kv_w3 { const char *z; const char *y; };
__attribute__((noinline)) static void w3_fill(struct kv_w3 *p, unsigned n)
{ unsigned i; for (i = 0; i < n; i++) { p[i].z = "x0"; p[i].y = "y0"; } }
static unsigned w3out_compute(void)
{
  struct kv_w3 a[32];
  volatile char pad[1200];
  unsigned i; int ok = 0;
  pad[0] = 1;
  w3_fill(a, 4);
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && w3out_len(a[i].z) > 0 && w3out_len(a[i].y) > 0) ok++;
  return (unsigned)ok + (unsigned)pad[0] - 1u;
}
#endif
