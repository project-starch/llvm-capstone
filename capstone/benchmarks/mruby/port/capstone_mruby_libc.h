/* Force-included before mruby's amalgamation: the handful of libc names it uses
 * that our freestanding headers do not carry.
 *
 * A header rather than edits to mruby, because none of these is a portability
 * defect -- mruby is entitled to expect <errno.h>, <math.h> and <stdlib.h> to
 * define them. Everything here is a gap in OUR environment.
 *
 * Kept minimal on purpose. Each name was added because a compile named it, not
 * because a real libc has it: an unused declaration here would be a silent claim
 * that something works.
 */
#ifndef CAPSTONE_MRUBY_LIBC_H
#define CAPSTONE_MRUBY_LIBC_H

/* errno values mruby's numeric conversions return. We have no errno variable
   worth the name in a domain, but mruby only ever compares against these. */
#ifndef ERANGE
#define ERANGE 34
#endif
#ifndef EDOM
#define EDOM 33
#endif

/* mrb_float is double here. HUGE_VAL is what strtod-style overflow returns, and
   mruby only ever assigns and compares it. __builtin_inf() gives the same value
   without needing a libm constant table. */
#ifndef HUGE_VAL
#define HUGE_VAL (__builtin_inf())
#endif

#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif
#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif

/* Declared, not defined: beebs' softfloat libm supplies the body. Without the
   declaration C99 rejects the call outright rather than guessing a signature. */
double hypot(double x, double y);
double log2(double x);
double cbrt(double x);

#endif /* CAPSTONE_MRUBY_LIBC_H */
