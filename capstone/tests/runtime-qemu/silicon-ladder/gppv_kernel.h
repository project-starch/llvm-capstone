#ifndef GPPV_KERNEL_H
#define GPPV_KERNEL_H
/* C-13 follow-up: ONE initializer path per rung, so a wrong checksum NAMES the broken
   mechanism instead of merely saying "gpstress is wrong".
   This variant: 512 B static const: a PRIVATE .L symbol, the SQLite-dominant shape */
static const unsigned g[128] = {
#define P8(i) (i)*13u+1u,(i)*13u+2u,(i)*13u+3u,(i)*13u+4u,(i)*13u+5u,(i)*13u+6u,(i)*13u+7u,(i)*13u+8u
#define P64(i) P8(i),P8(i+1),P8(i+2),P8(i+3),P8(i+4),P8(i+5),P8(i+6),P8(i+7)
  P64(0),P64(8)
};
static unsigned gppv_compute(void) {
  unsigned h = 2166136261u;
  for (int r=0;r<8;r++) for (int i=0;i<128;i++){ h^=g[i]; h*=16777619u; }
  return h;
}
#endif
