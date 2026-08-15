/* Force musl's <math.h> classification macros onto the double arm.
 *
 * THE TRAP. Every one of them is a ternary chain over sizeof(x) whose LAST arm
 * is the long-double one:
 *
 *     #define signbit(x) ( \
 *         sizeof(x) == sizeof(float)  ? (int)(__FLOAT_BITS(x)>>31) : \
 *         sizeof(x) == sizeof(double) ? (int)(__DOUBLE_BITS(x)>>63) : \
 *         __signbitl(x) )
 *
 * For a double argument the last arm is dead -- but only after constant
 * folding. At -O0 clang emits all three, so the link wants __signbitl,
 * __fpclassifyl and, to pass a double to them, __extenddftf2. None of those
 * exist or can exist on capstone64: every 128-bit float builtin fails to
 * compile (ISSUES.md C-20).
 *
 * This bites EVERY -O0 program compiled against musl here, not one workload. It
 * cost a link failure in vfprintf and again in mruby's numeric.c and fmt_fp.c,
 * which is why it is a header rather than a flag someone rediscovers.
 *
 * WHY A FORCED INCLUDE AND NOT -D ON THE COMMAND LINE. A -D is simply
 * overwritten: musl's math.h #defines the same names later. Included with
 * -include, this pulls math.h in FIRST and redefines the macros after it; the
 * .c file's own #include <math.h> is then a no-op against the include guard, so
 * these definitions survive.
 *
 * The clang builtins are type-generic and expand inline at every optimisation
 * level, with no libm call for any of them.
 */
#ifndef CAPSTONE_MATH_DOUBLE_H
#define CAPSTONE_MATH_DOUBLE_H

#include <math.h>

#undef signbit
#define signbit(x) __builtin_signbit(x)

#undef isfinite
#define isfinite(x) __builtin_isfinite(x)

#undef isinf
#define isinf(x) __builtin_isinf(x)

#undef isnan
#define isnan(x) __builtin_isnan(x)

#undef isnormal
#define isnormal(x) __builtin_isnormal(x)

#undef fpclassify
#define fpclassify(x)                                                          \
  __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, x)

#endif /* CAPSTONE_MATH_DOUBLE_H */
