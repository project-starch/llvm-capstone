#ifndef WBHI_H
#define WBHI_H
/* R-16 BLOB-BAND probe, at SQLite's geometry (build with DOMAIN_WINDOW=0x150000 and the
 * DEFAULT link-gpfree.ld). Geometry itself is now ruled out: wsq has SQLite's order-9 /
 * 0x150000 layout with a 144-byte blob and PASSES. The only attribute left that separates
 * f10 (blob 75120, ENTERS) from swa/strim (blob 84336/82592, STALL) is the blob band.
 * This puts ~88 KB of INITIALISED globals in the blob, above the stalling threshold.
 * The array is file-scope with an 8-multiple size: the generator's large-RO copy path
 * emits `lla <sym>` and rejects static/odd-sized symbols. Expect 4. */
char wbhi_arr[90112] = { 1 };
static unsigned wbhi_len(const char *s){unsigned n=0;while(s&&s[n])n++;return n;}
static unsigned wbhi_compute(void)
{
  struct kv { const char *z; const char *y; };
  struct kv a[8]; unsigned i; int ok = 0;
  wbhi_arr[0] = 1; wbhi_arr[90111] = 1;
  a[0].z="x0"; a[0].y="y0"; a[1].z="x0"; a[1].y="y0";
  a[2].z="x0"; a[2].y="y0"; a[3].z="x0"; a[3].y="y0";
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y && wbhi_len(a[i].z) > 0 && wbhi_len(a[i].y) > 0) ok++;
  return (unsigned)ok + (unsigned)(unsigned char)wbhi_arr[0] - 1u;
}
#endif
