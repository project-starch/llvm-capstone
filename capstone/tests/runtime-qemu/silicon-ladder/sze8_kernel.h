#ifndef SZE8_H
#define SZE8_H
/* Interp-glue discriminator: same COUNT (2 globals), same code, only the global SIZE
 * differs -- 8 bytes, an 8-byte multiple. Count is already
 * excluded: ri2 (two char[2]) FAILS under interp while wbi (three globals, one of them a
 * 64-byte array) PASSES. The glue's copy path is documented to care about 8-multiple sizes
 * (`size%8`, and the large-RO path rejects odd sizes), so size is the next suspect. */
static char s0[8] = { 1 };
static char s1[8] = { 1 };
static unsigned sze8_compute(void)
{
  unsigned r = 0;
  r += (unsigned)(unsigned char)s0[0];
  r += (unsigned)(unsigned char)s1[0];
  return r;
}
#endif
