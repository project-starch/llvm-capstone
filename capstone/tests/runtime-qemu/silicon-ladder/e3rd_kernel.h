#ifndef E3RD_KERNEL_H
#define E3RD_KERNEL_H
/* Bisecting DOWN from r14lp, which fails, toward the minimal probes that all PASS
 * (clp16: 16 ldc-from-gp in a loop; cgs8: ldc-from-gp + capability store to a computed
 * address in a loop; cst8: 8 capability stores; cdif8: 8 distinct slots). r14lp is the only
 * failing case left, so remove ONE of its remaining features at a time.
 * e3rd: r14lp with the READ-BACK unrolled (store loop kept) -- isolates the read loop. */
static unsigned e3rd_len(const char *s) { unsigned n = 0; while (s && s[n]) n++; return n; }
static unsigned e3rd_compute(void)
{
  struct kv { const char *z; const char *y; };
  struct kv a[64]; unsigned i; int ok = 0;
  for (i = 0; i < 4; i++) { a[i].z = "x0"; a[i].y = "y0"; }
  if (a[0].z && a[0].y && e3rd_len(a[0].z) > 0 && e3rd_len(a[0].y) > 0) ok++;
  if (a[1].z && a[1].y && e3rd_len(a[1].z) > 0 && e3rd_len(a[1].y) > 0) ok++;
  if (a[2].z && a[2].y && e3rd_len(a[2].z) > 0 && e3rd_len(a[2].y) > 0) ok++;
  if (a[3].z && a[3].y && e3rd_len(a[3].z) > 0 && e3rd_len(a[3].y) > 0) ok++;
  return (unsigned)ok;
}
#endif
