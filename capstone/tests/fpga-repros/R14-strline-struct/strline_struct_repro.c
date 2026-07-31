/* MINIMAL REPRODUCER -- straight-line init of a struct array with distinct string
 * constants wedges the Capstone CVA6 FPGA. See ISSUES.md R-14.
 *
 * Four variants, identical except for ONE variable each. Board-measured 2026-07-31 on
 * bitstream working-caplifive-captype-fixed.bit, gp-captable ABI, domain built by
 * capstone/benchmarks/sqlite/build-sqlite-silicon.sh (-O0 amalgam, -O1 string primitives).
 *
 *   A  16 distinct literals, STRAIGHT-LINE, struct{2 ptr}[64]  -> WEDGE (no return, no trap)
 *   B   4 distinct literals, straight-line + loop filler, same -> RETURNS 4, expected 16
 *   C  16 distinct literals, LOOP from a static table, same    -> RETURNS 16  (correct)
 *   D  16 distinct literals, STRAIGHT-LINE, flat ptr array[64] -> RETURNS 16  (correct)
 *
 * So the failure needs BOTH straight-line materialisation AND the struct element type.
 * B is the important one: it returns a WRONG VALUE rather than hanging, i.e. the same
 * construct corrupts silently at smaller scale.
 *
 * Each variant returns its count through the domain's result capability; the host prints it
 * as "SQ: obs=". A wedge prints nothing at all.
 */
typedef unsigned long usize;
extern int cap_strlen(const char *s);      /* indexed, linear-safe; see beebs_freestanding_string.c */

struct kv { const char *z; const char *y; };

int variant_A(void) {                      /* WEDGES on silicon */
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
  for (i=0;i<16;i++) if (a[i].z && a[i].y && cap_strlen(a[i].z)>0 && cap_strlen(a[i].y)>0) ok++;
  return ok;                               /* expect 16; observed: WEDGE */
}

int variant_B(void) {                      /* RETURNS 4 -- silent corruption */
  struct kv a[64]; unsigned i; int ok = 0;
  a[0].z="ltrim"; a[0].y="aaa0"; a[1].z="rtrim"; a[1].y="aaa1";
  a[2].z="trim";  a[2].y="aaa2"; a[3].z="max";   a[3].y="aaa3";
  for (i=4;i<64;i++){ a[i].z="filler"; a[i].y="fill"; }
  for (i=0;i<16;i++) if (a[i].z && a[i].y && cap_strlen(a[i].z)>0 && cap_strlen(a[i].y)>0) ok++;
  return ok;                               /* expect 16; observed 4 */
}

int variant_C(void) {                      /* correct */
  static const char *const tbl[16] = {
    "ltrim","rtrim","trim","max","min","typeof","length","instr",
    "substr","upper","lower","coalesce","hex","unhex","quote","replace" };
  struct kv a[64]; unsigned i; int ok = 0;
  for (i=0;i<16;i++){ a[i].z=tbl[i]; a[i].y="aaa0"; }
  for (i=16;i<64;i++){ a[i].z="filler"; a[i].y="fill"; }
  for (i=0;i<16;i++) if (a[i].z && a[i].y && cap_strlen(a[i].z)>0 && cap_strlen(a[i].y)>0) ok++;
  return ok;                               /* expect 16; observed 16 */
}

int variant_D(void) {                      /* correct */
  const char *f[64]; unsigned i; int ok = 0;
  f[0]="ltrim"; f[1]="rtrim"; f[2]="trim"; f[3]="max"; f[4]="min"; f[5]="typeof";
  f[6]="length"; f[7]="instr"; f[8]="substr"; f[9]="upper"; f[10]="lower";
  f[11]="coalesce"; f[12]="hex"; f[13]="unhex"; f[14]="quote"; f[15]="replace";
  for (i=16;i<64;i++) f[i]="filler";
  for (i=0;i<16;i++) if (f[i] && cap_strlen(f[i])>0) ok++;
  return ok;                               /* expect 16; observed 16 */
}
