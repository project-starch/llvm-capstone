#ifndef GPBG_KERNEL_H
#define GPBG_KERNEL_H
/* C-13 follow-up: ONE initializer path per rung, so a wrong checksum NAMES the broken
   mechanism instead of merely saying "gpstress is wrong".
   This variant: 2400 B initialized: past the generated glue's 2040 B per-global limit */
unsigned g[600] = {1,2,3,4,5,6,7,8,9,10};
static unsigned gpbg_compute(void) {
  unsigned h = 2166136261u;
  for (int r=0;r<8;r++) for (int i=0;i<600;i++){ h^=g[i]; h*=16777619u; }
  return h;
}
#endif
