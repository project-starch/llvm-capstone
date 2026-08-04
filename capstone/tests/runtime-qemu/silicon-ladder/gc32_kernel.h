#ifndef GC32_H
#define GC32_H
/* LAYOUT-SENSITIVITY SCAN -- the independent variable is the NUMBER OF GLOBALS.
 *
 * SQLite's behaviour moves with its global set: 183 carves entry-stalls, 178 hangs inside
 * strlen, 181 returns. Every attempt to instrument that image perturbed the same variable, so
 * instead vary it deliberately on a small image with byte-identical probe code.
 *
 * 32 separate initialised globals, each a struct holding a pointer to a string literal, then
 * a BOUNDED strlen over one of them. Bounded so the run always returns a number: a wrong
 * length is data, a hang is one bit.
 *   return = length * 1000 + (32 % 1000)
 */
struct gc32_kv { const char *z; long n; };
static const char *const gc32_lit[8] = {
  "alpha_one", "beta_two", "gamma_three", "delta_four",
  "epsilon_5", "zeta_six__", "eta_seven_", "theta_8___" };
static struct gc32_kv gc32_v0 = { gc32_lit[0], 0 };
static struct gc32_kv gc32_v1 = { gc32_lit[1], 1 };
static struct gc32_kv gc32_v2 = { gc32_lit[2], 2 };
static struct gc32_kv gc32_v3 = { gc32_lit[3], 3 };
static struct gc32_kv gc32_v4 = { gc32_lit[4], 4 };
static struct gc32_kv gc32_v5 = { gc32_lit[5], 5 };
static struct gc32_kv gc32_v6 = { gc32_lit[6], 6 };
static struct gc32_kv gc32_v7 = { gc32_lit[7], 7 };
static struct gc32_kv gc32_v8 = { gc32_lit[0], 8 };
static struct gc32_kv gc32_v9 = { gc32_lit[1], 9 };
static struct gc32_kv gc32_v10 = { gc32_lit[2], 10 };
static struct gc32_kv gc32_v11 = { gc32_lit[3], 11 };
static struct gc32_kv gc32_v12 = { gc32_lit[4], 12 };
static struct gc32_kv gc32_v13 = { gc32_lit[5], 13 };
static struct gc32_kv gc32_v14 = { gc32_lit[6], 14 };
static struct gc32_kv gc32_v15 = { gc32_lit[7], 15 };
static struct gc32_kv gc32_v16 = { gc32_lit[0], 16 };
static struct gc32_kv gc32_v17 = { gc32_lit[1], 17 };
static struct gc32_kv gc32_v18 = { gc32_lit[2], 18 };
static struct gc32_kv gc32_v19 = { gc32_lit[3], 19 };
static struct gc32_kv gc32_v20 = { gc32_lit[4], 20 };
static struct gc32_kv gc32_v21 = { gc32_lit[5], 21 };
static struct gc32_kv gc32_v22 = { gc32_lit[6], 22 };
static struct gc32_kv gc32_v23 = { gc32_lit[7], 23 };
static struct gc32_kv gc32_v24 = { gc32_lit[0], 24 };
static struct gc32_kv gc32_v25 = { gc32_lit[1], 25 };
static struct gc32_kv gc32_v26 = { gc32_lit[2], 26 };
static struct gc32_kv gc32_v27 = { gc32_lit[3], 27 };
static struct gc32_kv gc32_v28 = { gc32_lit[4], 28 };
static struct gc32_kv gc32_v29 = { gc32_lit[5], 29 };
static struct gc32_kv gc32_v30 = { gc32_lit[6], 30 };
static struct gc32_kv gc32_v31 = { gc32_lit[7], 31 };
static unsigned gc32_compute(void)
{
  int k = 0;
  const char *z;
  gc32_v0.n = 0;
  gc32_v1.n = 1;
  gc32_v2.n = 2;
  gc32_v3.n = 3;
  gc32_v4.n = 4;
  gc32_v5.n = 5;
  gc32_v6.n = 6;
  gc32_v7.n = 7;
  gc32_v8.n = 8;
  gc32_v9.n = 9;
  gc32_v10.n = 10;
  gc32_v11.n = 11;
  gc32_v12.n = 12;
  gc32_v13.n = 13;
  gc32_v14.n = 14;
  gc32_v15.n = 15;
  gc32_v16.n = 16;
  gc32_v17.n = 17;
  gc32_v18.n = 18;
  gc32_v19.n = 19;
  gc32_v20.n = 20;
  gc32_v21.n = 21;
  gc32_v22.n = 22;
  gc32_v23.n = 23;
  gc32_v24.n = 24;
  gc32_v25.n = 25;
  gc32_v26.n = 26;
  gc32_v27.n = 27;
  gc32_v28.n = 28;
  gc32_v29.n = 29;
  gc32_v30.n = 30;
  gc32_v31.n = 31;
  z = gc32_v16.z;                 /* pointer read back out of a global struct */
  while (k < 64 && z[k]) k++;
  return (unsigned)(k * 1000 + (32 % 1000));
}
#endif
