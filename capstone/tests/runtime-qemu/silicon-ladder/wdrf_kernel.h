#ifndef WDRF_H
#define WDRF_H
/* Links the cap-init NULL fault to the R-16 HANG, or shows they are different.
 * wbare NULL-TESTS its cap-init'd globals and so returns -62 (a value). SQLite DEREFERENCES
 * such globals unguarded, which is what a NULL capability turns into a fault. This does the
 * same: no null test, straight dereference of a capability-bearing initialised global.
 *   returns 65   -> the store worked here; wbare's NULL needs another explanation
 *   returns junk -> NULL/garbage capability, still no fault
 *   HANGS        -> the NULL store and the R-16 stall are the same defect
 * Expect 65. */
static char wdrf_data[64] = { 'A', 0 };
static char *wdrf_p = wdrf_data;
static unsigned wdrf_compute(void)
{
  return (unsigned)(unsigned char)wdrf_p[0];   /* UNGUARDED deref, as SQLite does */
}
#endif
