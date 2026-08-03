#ifndef AL_H
#define AL_H
/* DECIDES whether defects 1 and 2 are ONE defect: the byte-tail copy path.
 * The copy enters the byte tail on EITHER condition (start-gp-captable-interp.S record loop):
 *     size < 8            -> `blt a5, 8, 21f`
 *     blob_off % 8 != 0   -> `andi a6, t5, 7` ; `bnez a6, 21f`
 * With packed char[2] globals the blob offsets are 0,2,4: record 0 is aligned (and is the one
 * that works once the length is rounded), every later record is UNALIGNED and takes the tail.
 * Here the globals are still 2 bytes but forced to 8-byte ALIGNMENT, so every blob_off is a
 * multiple of 8 and no record can reach the tail via the alignment branch.
 *   777 -> one defect: the byte-tail path, entered via alignment for records 1+
 *   700 -> alignment is not the second trigger and they are genuinely two defects  */
static char al0[2] __attribute__((aligned(8))) = { 7, 0 };
static char al1[2] __attribute__((aligned(8))) = { 7, 0 };
static char al2[2] __attribute__((aligned(8))) = { 7, 0 };
static unsigned al_compute(void)
{
  return (unsigned)(unsigned char)al0[0]
       + 10u  * (unsigned)(unsigned char)al1[0]
       + 100u * (unsigned)(unsigned char)al2[0];
}
#endif
