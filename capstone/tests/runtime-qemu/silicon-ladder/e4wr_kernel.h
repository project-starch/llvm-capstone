#ifndef E4WR_KERNEL_H
#define E4WR_KERNEL_H
/* Bisecting DOWN from r14lp, which fails, toward the minimal probes that all PASS
 * (clp16: 16 ldc-from-gp in a loop; cgs8: ldc-from-gp + capability store to a computed
 * address in a loop; cst8: 8 capability stores; cdif8: 8 distinct slots). r14lp is the only
 * failing case left, so remove ONE of its remaining features at a time.
 * e4wr: r14lp with the STORE loop unrolled (read loop kept) -- the mirror of e3rd. */
static unsigned e4wr_len(const char *s) { unsigned n = 0; while (s && s[n]) n++; return n; }
static unsigned e4wr_compute(void)
{
  struct kv { const char *z; const char *y; };
  struct kv a[64]; unsigned i; int ok = 0;
  a[0].z = "x0"; a[0].y = "y0";  a[1].z = "x0"; a[1].y = "y0";
  a[2].z = "x0"; a[2].y = "y0";  a[3].z = "x0"; a[3].y = "y0";
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && e4wr_len(a[i].z) > 0 && e4wr_len(a[i].y) > 0) ok++;
  return (unsigned)ok;
}
#endif
