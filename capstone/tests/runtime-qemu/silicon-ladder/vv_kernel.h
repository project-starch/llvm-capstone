#ifndef VV_H
#define VV_H
/* Return the ACTUAL bytes of three distinct globals, not a pass/fail count.
 * Cap-table slots are already proven correct and distinct (pk = 117), so the carve is fine and
 * only the copied CONTENT can be wrong. Prediction under "the blob source pointer does not
 * advance per record": every record copies record 0's bytes, so all three read 1.
 *   correct  -> 1 + 10*2 + 100*3 = 321
 *   all r0   -> 1 + 10*1 + 100*1 = 111   <- source pointer never advances
 *   zeros    -> 1 (or 0)                 <- nothing copied for records 1,2  */
static char vv0[2] = { 1, 0 };
static char vv1[2] = { 2, 0 };
static char vv2[2] = { 3, 0 };
static unsigned vv_compute(void)
{
  return (unsigned)(unsigned char)vv0[0]
       + 10u  * (unsigned)(unsigned char)vv1[0]
       + 100u * (unsigned)(unsigned char)vv2[0];
}
#endif
