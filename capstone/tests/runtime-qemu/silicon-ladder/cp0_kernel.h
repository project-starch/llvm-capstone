#ifndef CP0_H
#define CP0_H
/* CODE-LAYOUT sensitivity scan. Global COUNT was excluded: gc4..gc200 (8..208 carves) all
 * returned correctly. But an opaque clamp whose descriptor table was byte-identical to the
 * unclamped build still flipped SQLite's stage 11 -- a stage that never calls the modified
 * function. So the variable under test here is CODE LAYOUT, with the globals and the probe
 * held byte-identical across draws.
 *
 * 1 never-called functions shift .text by roughly 0 KiB. The probe is the same bounded
 * strlen over a pointer read out of a global struct.
 *   return = length * 1000 + (0 % 1000)
 */
struct cp0_kv { const char *z; long n; };
static const char *const cp0_lit[8] = {
  "alpha_one", "beta_two", "gamma_three", "delta_four",
  "epsilon_5", "zeta_six__", "eta_seven_", "theta_8___" };
static struct cp0_kv cp0_v = { cp0_lit[2], 0 };
__attribute__((noinline,used)) static long cp0_f0(long x){ long a=x+0; a^=a<<3; a+=a>>2; a*=3; a^=a<<5; a+=a>>1; a*=7; a^=a<<2; return a; }
static unsigned cp0_compute(void)
{
  int k = 0;
  const char *z;
  cp0_v.n = 1;
  z = cp0_v.z;
  while (k < 64 && z[k]) k++;
  return (unsigned)(k * 1000 + (0 % 1000));
}
#endif
