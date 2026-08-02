/* R-14 variant B as a QEMU-gated ladder rung (2026-08-02).

   Reduced verbatim from tests/fpga-repros/R14-strline-struct/strline_struct_repro.c.
   Board-measured 2026-07-31: this RETURNS 4 where 16 is correct -- it does NOT wedge.

   That makes it the most useful variant by a wide margin: the failure is a NUMBER, not a
   hang, so it can be bisected without losing the board session (the project's own
   "make every run RETURN" rule). Same struct as variant A, differing only in how many
   entries are materialised straight-line: 4 here vs 16 in A, remainder loop-filled.
   The four straight-line entries pass and the twelve loop-assigned ones fail.

   C-16 (memset tail-padding tag strip) does NOT apply: `struct kv` is two capabilities =
   32 bytes with no tail padding, and the array is uninitialised then assigned
   element-by-element, so no initialiser memset exists at all.  Expect 16.  */
struct kv { const char *z; const char *y; };

static unsigned cap_strlen(const char *s)
{
  unsigned n = 0;
  while (s && s[n]) n++;
  return n;
}

static int variant_B(void)
{
  struct kv a[64]; unsigned i; int ok = 0;
  a[0].z="ltrim"; a[0].y="aaa0"; a[1].z="rtrim"; a[1].y="aaa1";
  a[2].z="trim";  a[2].y="aaa2"; a[3].z="max";   a[3].y="aaa3";
  for (i=4;i<64;i++){ a[i].z="filler"; a[i].y="fill"; }
  for (i=0;i<16;i++)
    if (a[i].z && a[i].y && cap_strlen(a[i].z)>0 && cap_strlen(a[i].y)>0) ok++;
  return ok;                               /* expect 16; board observed 4 */
}

void domain_main(unsigned *res, unsigned func) { (void)func; *res = (unsigned)variant_B(); }
