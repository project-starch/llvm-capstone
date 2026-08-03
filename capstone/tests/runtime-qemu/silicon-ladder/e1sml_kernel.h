#ifndef E1SML_KERNEL_H
#define E1SML_KERNEL_H
/* Bisecting DOWN from r14lp, which fails, toward the minimal probes that all PASS
 * (clp16: 16 ldc-from-gp in a loop; cgs8: ldc-from-gp + capability store to a computed
 * address in a loop; cst8: 8 capability stores; cdif8: 8 distinct slots). r14lp is the only
 * failing case left, so remove ONE of its remaining features at a time.
 * e1sml: r14lp with a[8] instead of a[64] -- isolates the 2 KB stack array SIZE. */
static unsigned e1sml_len(const char *s) { unsigned n = 0; while (s && s[n]) n++; return n; }
static unsigned e1sml_compute(void)
{
  struct kv { const char *z; const char *y; };
  struct kv a[8]; unsigned i; int ok = 0;
  for (i = 0; i < 4; i++) { a[i].z = "x0"; a[i].y = "y0"; }
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && e1sml_len(a[i].z) > 0 && e1sml_len(a[i].y) > 0) ok++;
  return (unsigned)ok;
}
#endif
