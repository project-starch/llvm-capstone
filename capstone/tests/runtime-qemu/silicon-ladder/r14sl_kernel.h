#ifndef R14SL_KERNEL_H
#define R14SL_KERNEL_H
/* R-14 discriminator, arm SL: four struct entries assigned STRAIGHT-LINE.
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
struct kv_sl { const char *z; const char *y; };
static unsigned r14sl_len(const char *s) { unsigned n = 0; while (s && s[n]) n++; return n; }
static unsigned r14sl_compute(void)
{
  struct kv_sl a[64]; unsigned i; int ok = 0;
  a[0].z = "x0"; a[0].y = "y0";
  a[1].z = "x0"; a[1].y = "y0";
  a[2].z = "x0"; a[2].y = "y0";
  a[3].z = "x0"; a[3].y = "y0";
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && r14sl_len(a[i].z) > 0 && r14sl_len(a[i].y) > 0) ok++;
  return (unsigned)ok;                      /* expect 4 */
}
#endif
