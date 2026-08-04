#ifndef GC4_H
#define GC4_H
/* LAYOUT-SENSITIVITY SCAN -- the independent variable is the NUMBER OF GLOBALS.
 *
 * SQLite's behaviour moves with its global set: 183 carves entry-stalls, 178 hangs inside
 * strlen, 181 returns. Every attempt to instrument that image perturbed the same variable, so
 * instead vary it deliberately on a small image with byte-identical probe code.
 *
 * 4 separate initialised globals, each a struct holding a pointer to a string literal, then
 * a BOUNDED strlen over one of them. Bounded so the run always returns a number: a wrong
 * length is data, a hang is one bit.
 *   return = length * 1000 + (4 % 1000)
 */
struct gc4_kv { const char *z; long n; };
static const char *const gc4_lit[8] = {
  "alpha_one", "beta_two", "gamma_three", "delta_four",
  "epsilon_5", "zeta_six__", "eta_seven_", "theta_8___" };
static struct gc4_kv gc4_v0 = { gc4_lit[0], 0 };
static struct gc4_kv gc4_v1 = { gc4_lit[1], 1 };
static struct gc4_kv gc4_v2 = { gc4_lit[2], 2 };
static struct gc4_kv gc4_v3 = { gc4_lit[3], 3 };
static unsigned gc4_compute(void)
{
  int k = 0;
  const char *z;
  gc4_v0.n = 0;
  gc4_v1.n = 1;
  gc4_v2.n = 2;
  gc4_v3.n = 3;
  z = gc4_v2.z;                 /* pointer read back out of a global struct */
  while (k < 64 && z[k]) k++;
  return (unsigned)(k * 1000 + (4 % 1000));
}
#endif
