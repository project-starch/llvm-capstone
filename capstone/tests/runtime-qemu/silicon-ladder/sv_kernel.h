#ifndef SV_H
#define SV_H
/* Discriminates SOURCE-pointer vs DESTINATION/counter for defect 2.
 * Established: cap-table slots are valid and DISTINCT (pk=117), so destinations are correct;
 * and with the copy-length fix exactly ONE record's content lands (vf=300, only the
 * first-PROCESSED record).
 * Here all three globals hold the SAME value. If the blob SOURCE pointer never advances, every
 * record copies record 0's bytes -- which are now identical to its own -- so ALL THREE become
 * correct and this returns 777. If the destination or the loop counter is at fault, the value
 * being identical changes nothing and only one lands (7, 70 or 700).
 *   777 -> source pointer is the defect
 *   7 / 70 / 700 -> destination or counter is the defect  */
static char sv0[2] = { 7, 0 };
static char sv1[2] = { 7, 0 };
static char sv2[2] = { 7, 0 };
static unsigned sv_compute(void)
{
  return (unsigned)(unsigned char)sv0[0]
       + 10u  * (unsigned)(unsigned char)sv1[0]
       + 100u * (unsigned)(unsigned char)sv2[0];
}
#endif
