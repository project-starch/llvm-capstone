#ifndef WCAP_H
#define WCAP_H
/* R-16: the LAST untested axis -- __capstone_cap_init, which the ladder has NEVER EXECUTED.
 * Every rung has .capstone_cap_init at size 0; every SQLite image has it non-empty. It runs at
 * DOMAIN ENTRY, building capability-bearing globals, which is exactly where R-16 stalls -- the
 * classic presence-vs-execution trap.
 * Ruled out by now, each with a one-variable pair: image size (rz1m), carve count (rc192),
 * their conjunction (rzc1m), dom_data geometry (wsq, SQLite's order-9 / 0x150000 layout), blob
 * size (wbhi, blob 90320 > the 84336 that stalls), and the loader (strim stalls under lpc too).
 * A capability-bearing INITIALISED global is what forces cap_init to exist and run. Expect 4. */
static char wcap_data[64] = { 'A', 0 };
static char *wcap_ptr = wcap_data;            /* capability-bearing initialised global */
static char *wcap_ptr2 = wcap_data + 16;
static unsigned wcap_compute(void)
{
  unsigned n = 0;
  if (wcap_ptr) n += (unsigned)(unsigned char)wcap_ptr[0];      /* 'A' = 65 */
  if (wcap_ptr2) n += 1u;
  return n - 62u;                                               /* 65 + 1 - 62 = 4 */
}
#endif
