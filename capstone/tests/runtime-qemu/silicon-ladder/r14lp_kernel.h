#ifndef R14LP_KERNEL_H
#define R14LP_KERNEL_H
/* R-14 discriminator, arm LP: the same four entries assigned in a LOOP.
 *
 * Paired with r14lp, which is byte-for-byte the same computation except that the
 * four entries are assigned in a LOOP. Everything else is held constant on purpose:
 * the SAME two literals in every field (so string-constant distinctness cannot
 * matter -- already refuted, but keep it controlled), the SAME number of capability
 * stores, the SAME array slots touched, and the SAME read-back loop.
 *
 * The only difference is HOW the store address is formed: immediate offsets here,
 * a computed (loop-variable) address in r14lp. r14b_app.c records that its four
 * straight-line entries PASS while its twelve loop-assigned ones FAIL, which makes
 * this the axis none of the 2026-08-03 arms tested. Expect 4. */
struct kv_lp { const char *z; const char *y; };
static unsigned r14lp_len(const char *s) { unsigned n = 0; while (s && s[n]) n++; return n; }
static unsigned r14lp_compute(void)
{
  struct kv_lp a[64]; unsigned i; int ok = 0;
  for (i = 0; i < 4; i++) { a[i].z = "x0"; a[i].y = "y0"; }
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && r14lp_len(a[i].z) > 0 && r14lp_len(a[i].y) > 0) ok++;
  return (unsigned)ok;                      /* expect 4 */
}
#endif
