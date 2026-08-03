#ifndef F2NOP_KERNEL_H
#define F2NOP_KERNEL_H
/* FRAME-SIZE hypothesis. Measured: a[8] (256 B) and a[64]-with-one-cap (1024 B) PASS,
 * a[64]-with-two-caps (2048 B) FAILS -- and it fails with the store loop unrolled AND with
 * the read loop unrolled, so loops are irrelevant. 2048 is exactly the RISC-V 12-bit signed
 * immediate limit: past it the compiler stops using `cincoffsetimm off(s0)` and computes
 * frame addresses with `lui` + REGISTER-form `cincoffset`, which is the two-step form seen
 * in r14lp's disassembly. If the FRAME is what matters, a small array plus dead padding
 * must fail too, and the identical array without padding must pass.
 * f2nop: the SAME a[32] with NO padding -> frame < 2048. The control for f1pad. */
static unsigned f2nop_len(const char *s) { unsigned n = 0; while (s && s[n]) n++; return n; }
static unsigned f2nop_compute(void)
{
  struct kv { const char *z; const char *y; };
  struct kv a[32];
  unsigned i; int ok = 0;
  for (i = 0; i < 4; i++) { a[i].z = "x0"; a[i].y = "y0"; }
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && f2nop_len(a[i].z) > 0 && f2nop_len(a[i].y) > 0) ok++;
  return (unsigned)ok;
}
#endif
