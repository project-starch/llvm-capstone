#ifndef ZOFF_H
#define ZOFF_H
/* Tests the predicted compiler fix for R-14: k1200's exact shape, but each capability store
 * goes through an explicitly materialised address so the `stc` immediate is ZERO instead of
 * folding the field offset (0x10) into it. k1200 FAILS; if zoff PASSES with identical work,
 * the fix is a codegen constraint -- never fold a field offset into an `stc` immediate when
 * the base register is register-derived. Expect 4. */
static unsigned zoff_len(const char *s){unsigned n=0;while(s&&s[n])n++;return n;}
static unsigned zoff_compute(void)
{
  struct kv { const char *z; const char *y; };
  struct kv a[32];
  volatile char pad[1200];
  unsigned i; int ok = 0;
  pad[0] = 1;
  for (i = 0; i < 4; i++) {
    const char *volatile *pz = &a[i].z;  *pz = "x0";
    const char *volatile *py = &a[i].y;  *py = "y0";
  }
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && zoff_len(a[i].z) > 0 && zoff_len(a[i].y) > 0) ok++;
  return (unsigned)ok + (unsigned)pad[0] - 1u;
}
#endif
