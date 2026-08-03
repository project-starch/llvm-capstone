#ifndef R14B_KERNEL_H
#define R14B_KERNEL_H
/* Board-perf kernel form of r14b_app.c's variant_B, so the KNOWN-FAILING case rides the
   same batch as the r14sl/r14lp discriminator and is measured under identical conditions.
   Board-measured 2026-07-31: RETURNS 4 where 16 is correct -- a number, not a hang. The
   file's own note records that the four STRAIGHT-LINE entries pass and the twelve
   LOOP-ASSIGNED ones fail, which is what r14sl/r14lp isolate. Expect 16. */
struct kv_b { const char *z; const char *y; };
static unsigned r14b_len(const char *s) { unsigned n = 0; while (s && s[n]) n++; return n; }
static unsigned r14b_compute(void)
{
  struct kv_b a[64]; unsigned i; int ok = 0;
  a[0].z="ltrim"; a[0].y="aaa0"; a[1].z="rtrim"; a[1].y="aaa1";
  a[2].z="trim";  a[2].y="aaa2"; a[3].z="max";   a[3].y="aaa3";
  for (i=4;i<64;i++){ a[i].z="filler"; a[i].y="fill"; }
  for (i=0;i<16;i++)
    if (a[i].z && a[i].y && r14b_len(a[i].z)>0 && r14b_len(a[i].y)>0) ok++;
  return (unsigned)ok;                      /* expect 16 */
}
#endif
