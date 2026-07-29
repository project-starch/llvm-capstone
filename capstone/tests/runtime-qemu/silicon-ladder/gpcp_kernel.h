#ifndef GPCP_KERNEL_H
#define GPCP_KERNEL_H
/* C-13 follow-up: ONE initializer path per rung, so a wrong checksum NAMES the broken
   mechanism instead of merely saying "gpstress is wrong".
   This variant: 256 B initialized, size%8==0: the 8-byte bulk COPY loop */
unsigned g[64] = {
#define C8(i) (i)*11u+1u,(i)*11u+2u,(i)*11u+3u,(i)*11u+4u,(i)*11u+5u,(i)*11u+6u,(i)*11u+7u,(i)*11u+8u
  C8(0),C8(1),C8(2),C8(3),C8(4),C8(5),C8(6),C8(7)
};
static unsigned gpcp_compute(void) {
  unsigned h = 2166136261u;
  for (int r=0;r<8;r++) for (int i=0;i<64;i++){ h^=g[i]; h*=16777619u; }
  return h;
}
#endif
