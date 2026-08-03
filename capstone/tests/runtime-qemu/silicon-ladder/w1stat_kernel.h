#ifndef W1STAT_H
#define W1STAT_H
/* WORKAROUND CANDIDATE for R-14. Baseline: k1200 (struct a[32] + 1200 B dead pad -> frame
 * 4576 B, 13 lui-addressing sites) FAILS on silicon while k800 (800 B pad, frame 3776, 0
 * lui sites) passes. The fault needs a big lui-addressed frame, so a workaround must stop
 * the capability stores happening in one.
 * w1stat: the array becomes a file-scope STATIC, so it leaves the frame entirely. This is the
 * minimal-repro analogue of SQLITE_STATIC_BUILTINS=1. The 1200 B pad is KEPT, so if the frame
 * alone were the trigger this would still fail -- it tests moving the STORE TARGET out. */
static unsigned w1stat_len(const char *s){unsigned n=0;while(s&&s[n])n++;return n;}
struct kv_w1 { const char *z; const char *y; };
static struct kv_w1 w1_a[32];
static unsigned w1stat_compute(void)
{
  volatile char pad[1200];
  unsigned i; int ok = 0;
  pad[0] = 1;
  for (i = 0; i < 4; i++) { w1_a[i].z = "x0"; w1_a[i].y = "y0"; }
  for (i = 0; i < 4; i++)
    if (w1_a[i].z && w1_a[i].y && w1stat_len(w1_a[i].z) > 0 && w1stat_len(w1_a[i].y) > 0) ok++;
  return (unsigned)ok + (unsigned)pad[0] - 1u;
}
#endif
