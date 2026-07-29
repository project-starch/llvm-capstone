#ifndef GPSZ_KERNEL_H
#define GPSZ_KERNEL_H
/* C-13 follow-up: ONE initializer path per rung, so a wrong checksum NAMES the broken
   mechanism instead of merely saying "gpstress is wrong".
   This variant: zero-init (.bss): blob_off == -1, pure zero-fill path */
static unsigned g[64];
static unsigned gpsz_compute(void) {
  unsigned h = 2166136261u;
  for (int i=0;i<64;i++) g[i] = (unsigned)i*3u+7u;
  for (int r=0;r<8;r++) for (int i=0;i<64;i++){ h^=g[i]; h*=16777619u; }
  return h;
}
#endif
