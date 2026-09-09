/* __floatdidf without globals, for the single-module silicon build.
 *
 * WHY. compiler-rt's floatdidf.c is the ONLY float builtin SQLite pulls in that owns
 * data: two file-scope constants, __floatdidf.twop32 and .twop52. Under
 * -capstone-gp-captable a second translation unit that owns globals collides with the
 * first on the single gp cap-table (per-module positional indices), so it would have
 * to be amalgamated -- and it cannot be: SQLITE_OMIT_FLOATING_POINT makes SQLite
 * `#define double sqlite_int64`, which poisons compiler-rt's int_types.h the moment it
 * lands in the same TU ("typedef redefinition, 'long' vs 'long long'").
 *
 * So: compiler-rt's algorithm verbatim, with the two constants materialized as LOCALS
 * from their IEEE-754 bit patterns instead of file-scope data. No globals means no
 * cap-table slot means no collision, so it stays a separate object like the other
 * builtins.
 *
 * Multiply and add only -- deliberately no division. A first version used divides and
 * pulled in an undefined __divdf3, which is not in the builtin set this domain links.
 *
 * `dbl` via a typedef rather than the `double` keyword: this file is compiled with the
 * SQLite defines on the command line, so the `double` macro is live here too.
 */
typedef long long              di_int_t;
typedef unsigned long long     du_int_t;
typedef int                    si_int_t;

typedef union { du_int_t u; double d; } dbl_bits;

double __floatdidf(di_int_t a) {
  const dbl_bits twop52 = { .u = 0x4330000000000000ULL };   /* 2^52 */
  const dbl_bits twop32 = { .u = 0x41F0000000000000ULL };   /* 2^32 */

  if (a == 0)
    return 0.0;

  dbl_bits low = twop52;
  const double high = (double)(si_int_t)(a >> 32) * twop32.d;
  low.u |= (du_int_t)a & 0x00000000FFFFFFFFULL;
  return (high - twop52.d) + low.d;
}
