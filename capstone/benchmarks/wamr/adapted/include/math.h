/* Freestanding <math.h> for WAMR. The classification macros go to builtins, which
   the backend lowers without a libm call; the transcendentals are declared and
   come from beebs_softfloat_libm.c where a build needs them. */
#ifndef CAPSTONE_WAMR_MATH_H
#define CAPSTONE_WAMR_MATH_H
#define isnan(x)     __builtin_isnan(x)
#define isinf(x)     __builtin_isinf(x)
#define isfinite(x)  __builtin_isfinite(x)
#define signbit(x)   __builtin_signbit(x)
#define NAN          __builtin_nanf("")
#define INFINITY     __builtin_inff()
double fabs(double); double sqrt(double); double floor(double); double ceil(double);
double trunc(double); double rint(double); double pow(double, double);
float fabsf(float); float sqrtf(float); float floorf(float); float ceilf(float);
float truncf(float); float rintf(float);
double copysign(double, double); float copysignf(float, float);
#endif
