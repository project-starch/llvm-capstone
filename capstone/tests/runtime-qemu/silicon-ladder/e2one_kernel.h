#ifndef E2ONE_KERNEL_H
#define E2ONE_KERNEL_H
/* Bisecting DOWN from r14lp, which fails, toward the minimal probes that all PASS
 * (clp16: 16 ldc-from-gp in a loop; cgs8: ldc-from-gp + capability store to a computed
 * address in a loop; cst8: 8 capability stores; cdif8: 8 distinct slots). r14lp is the only
 * failing case left, so remove ONE of its remaining features at a time.
 * e2one: r14lp with ONE capability per entry (no .y) -- isolates two-fields-per-entry. */
static unsigned e2one_len(const char *s) { unsigned n = 0; while (s && s[n]) n++; return n; }
static unsigned e2one_compute(void)
{
  struct kv { const char *z; };
  struct kv a[64]; unsigned i; int ok = 0;
  for (i = 0; i < 4; i++) { a[i].z = "x0"; }
  for (i = 0; i < 4; i++)
    if (a[i].z && e2one_len(a[i].z) > 0) ok++;
  return (unsigned)ok;
}
#endif
