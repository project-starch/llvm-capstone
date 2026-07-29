#ifndef GPTL_KERNEL_H
#define GPTL_KERNEL_H
/* C-13 follow-up: ONE initializer path per rung, so a wrong checksum NAMES the broken
   mechanism instead of merely saying "gpstress is wrong".
   This variant: 13 B initialized, size%8 != 0: exercises the BYTE TAIL (lb/sb) */
unsigned char g[13] = {3,1,4,1,5,9,2,6,5,3,5,8,9};
static unsigned gptl_compute(void) {
  unsigned h = 2166136261u;
  for (int r=0;r<8;r++) for (int i=0;i<13;i++){ h^=g[i]; h*=16777619u; }
  return h;
}
#endif
