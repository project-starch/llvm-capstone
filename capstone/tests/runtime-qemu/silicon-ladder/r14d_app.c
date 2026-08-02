/* R-14 variant D as a QEMU-gated ladder rung (2026-08-02) -- CONTROL.
   Flat `const char *[64]`, straight-line, 16 distinct literals. Board result: CORRECT (16).
   It is the control that isolates "struct element type" as the necessary ingredient:
   same straight-line materialisation as variant A, but no struct. Expect 16. */
static unsigned cap_strlen(const char *s)
{
  unsigned n = 0;
  while (s && s[n]) n++;
  return n;
}

static int variant_D(void)
{
  const char *f[64]; unsigned i; int ok = 0;
  f[0]="ltrim"; f[1]="rtrim"; f[2]="trim"; f[3]="max"; f[4]="min"; f[5]="typeof";
  f[6]="length"; f[7]="instr"; f[8]="substr"; f[9]="upper"; f[10]="lower";
  f[11]="coalesce"; f[12]="hex"; f[13]="unhex"; f[14]="quote"; f[15]="replace";
  for (i=16;i<64;i++) f[i]="filler";
  for (i=0;i<16;i++) if (f[i] && cap_strlen(f[i])>0) ok++;
  return ok;                               /* expect 16 */
}

void domain_main(unsigned *res, unsigned func) { (void)func; *res = (unsigned)variant_D(); }
