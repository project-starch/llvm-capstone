#ifndef R14HL_KERNEL_H
#define R14HL_KERNEL_H
/* R-14 discriminator, arm HL: the LOOP of r14lp, but with the cap-table loads HOISTED.
 *
 * Separates the two mechanisms r14sl-vs-r14lp leaves tangled. r14lp differs from r14sl in
 * TWO ways at once: (a) the store address is computed per iteration, and (b) the literal is
 * re-loaded from the SAME cap-table slot every iteration (`ldc a2, 0x0(gp)` sits inside the
 * loop, so it executes 4x instead of once).
 *
 * This arm keeps (a) -- identical loop, identical computed addresses -- and changes (b).
 * CAREFUL, the source-level reading is wrong: at -O0 these locals are SPILLED, so the loop
 * still executes the same number of dynamic ldc's as r14lp (2 per iteration). What actually
 * differs is WHICH memory is re-read -- a STACK slot here (`ldc a0, 0x0(a0)`) versus the
 * CAP-TABLE (`ldc a2, 0x0(gp)`) in r14lp. Do not describe this arm as "ldc hoisted". So:
 *    HL passes  => repeated `ldc` from one slot is the trigger, NOT the computed address.
 *    HL fails   => the computed address is the trigger, and the ldc placement is innocent.
 *
 * Why (b) is a real candidate: capstone-ariane/CLAUDE.md documents "after an LDC that loads a
 * LINEAR capability, the source memory location is CLEARED to prevent aliasing" -- so a second
 * ldc of the same slot would read a cleared slot. It also matches r14b_app.c exactly: its four
 * straight-line entries (one load per literal) pass, its twelve loop-assigned ones (the same
 * "filler"/"fill" reloaded every iteration) fail.
 * NOTE the counter-argument, which is why this is a test and not a conclusion:
 * build-sqlite-silicon.sh states cap-table entries are ALREADY NONLIN, and clearing is
 * documented for LINEAR only. Expect 4. */
struct kv_hl { const char *z; const char *y; };
static unsigned r14hl_len(const char *s) { unsigned n = 0; while (s && s[n]) n++; return n; }
static unsigned r14hl_compute(void)
{
  struct kv_hl a[64]; unsigned i; int ok = 0;
  const char *z0 = "x0";                    /* loaded ONCE, before the loop */
  const char *y0 = "y0";
  for (i = 0; i < 4; i++) { a[i].z = z0; a[i].y = y0; }
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && r14hl_len(a[i].z) > 0 && r14hl_len(a[i].y) > 0) ok++;
  return (unsigned)ok;                      /* expect 4 */
}
#endif
