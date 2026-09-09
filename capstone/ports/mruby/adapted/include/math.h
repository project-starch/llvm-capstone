#pragma once
#include <stddef.h>

typedef float float_t;
typedef double double_t;

#define INFINITY (__builtin_inf())
#define NAN (__builtin_nan(""))

#define isfinite(x) __builtin_isfinite(x)
#define isinf(x) __builtin_isinf(x)
#define isnan(x) __builtin_isnan(x)
#define signbit(x) __builtin_signbit(x)

double acos(double);
double acosh(double);
double asin(double);
double asinh(double);
double atan(double);
double atan2(double, double);
double atanh(double);
double ceil(double);
double copysign(double, double);
double cos(double);
double cosh(double);
double erf(double);
double erfc(double);
double exp(double);
double expm1(double);
double fabs(double);
double floor(double);
double fmod(double, double);
double frexp(double, int *);
double ldexp(double, int);
double lgamma(double);
double log(double);
double log10(double);
double log1p(double);
double modf(double, double *);
double nan(const char *);
double nearbyint(double);
double pow(double, double);
double rint(double);
double round(double);
double scalbn(double, int);
double sin(double);
double sinh(double);
double sqrt(double);
double tan(double);
double tanh(double);
double tgamma(double);
double trunc(double);
