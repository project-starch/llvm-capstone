/* R-14 variant A as a QEMU-gated ladder rung (2026-08-02).

   Reduced verbatim from tests/fpga-repros/ARCHIVED/R14-strline-struct/strline_struct_repro.c.
   Board-measured 2026-07-31: this WEDGES on silicon while variants C and D return 16.

   Recorded here because C-16 (the memset tail-padding tag-strip) provably does NOT explain
   it: `struct kv` is two capabilities = 32 bytes with NO tail padding, and the array is
   declared uninitialised and assigned element-by-element, so no aggregate initialiser and no
   memset are involved. If this rung passes under QEMU with the C-16 fix in place, R-14 is a
   genuinely silicon-only defect and is NOT a duplicate of C-16.  Expect 16.  */
struct kv { const char *z; const char *y; };

static unsigned cap_strlen(const char *s)
{
  unsigned n = 0;
  while (s && s[n]) n++;
  return n;
}

static int variant_A(void)
{
  struct kv a[64]; unsigned i; int ok = 0;
  a[0].z="ltrim";  a[0].y="aaa0";   a[1].z="rtrim";  a[1].y="aaa1";
  a[2].z="trim";   a[2].y="aaa2";   a[3].z="max";    a[3].y="aaa3";
  a[4].z="min";    a[4].y="aaa4";   a[5].z="typeof"; a[5].y="aaa5";
  a[6].z="length"; a[6].y="aaa6";   a[7].z="instr";  a[7].y="aaa7";
  a[8].z="substr"; a[8].y="aaa8";   a[9].z="upper";  a[9].y="aaa9";
  a[10].z="lower"; a[10].y="aab0";  a[11].z="coalesce"; a[11].y="aab1";
  a[12].z="hex";   a[12].y="aab2";  a[13].z="unhex"; a[13].y="aab3";
  a[14].z="quote"; a[14].y="aab4";  a[15].z="replace"; a[15].y="aab5";
  for (i=16;i<64;i++){ a[i].z="filler"; a[i].y="fill"; }
  for (i=0;i<16;i++)
    if (a[i].z && a[i].y && cap_strlen(a[i].z)>0 && cap_strlen(a[i].y)>0) ok++;
  return ok;                               /* expect 16 */
}

void domain_main(unsigned *res, unsigned func) { (void)func; *res = (unsigned)variant_A(); }
