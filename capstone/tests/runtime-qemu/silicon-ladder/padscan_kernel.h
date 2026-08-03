#ifndef P_H
#define P_H
static unsigned padscan_compute(void)
{
  struct kv { const char *z; const char *y; };
  struct kv a[32];
  volatile char pad[2000];
  unsigned i; int ok = 0;
  pad[0] = 1;
  for (i = 0; i < 4; i++) { a[i].z = "x0"; a[i].y = "y0"; }
  for (i = 0; i < 4; i++)
    if (a[i].z && a[i].y) ok++;
  return (unsigned)ok;
}
#endif
