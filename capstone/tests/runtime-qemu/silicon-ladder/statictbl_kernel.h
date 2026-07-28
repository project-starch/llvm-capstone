#ifndef STATICTBL_KERNEL_H
#define STATICTBL_KERNEL_H
/* Decisive test for the descriptor-driven glue: a STATIC const table >256 B.
   Static => private (.L) symbol => the generated glue's copy path REJECTS it and
   must unroll ~25 bytes of .text per byte of data. This is the SQLite shape. */
static const unsigned st_tab[256] = {
#define R8(i) (i)*7u+1u,(i)*7u+2u,(i)*7u+3u,(i)*7u+4u,(i)*7u+5u,(i)*7u+6u,(i)*7u+7u,(i)*7u+8u
#define R64(i) R8(i),R8(i+1),R8(i+2),R8(i+3),R8(i+4),R8(i+5),R8(i+6),R8(i+7)
  R64(0), R64(8), R64(16), R64(24)
};
static unsigned st_compute(void){
  unsigned h=2166136261u;
  for(int r=0;r<64;r++) for(int i=0;i<256;i++){ h^=st_tab[(i+r)&255]; h*=16777619u; }
  return h;
}
#endif
