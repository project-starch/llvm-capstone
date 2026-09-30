#ifndef _MATH_H
#define _MATH_H 1
double pow(double, double);
double floor(double);
double ceil(double);
double fmod(double, double);
double fabs(double);
double log(double);
double log10(double);
double exp(double);
double sqrt(double);
int finite(double);
int isnan(double);
int isinf(double);
#define HUGE_VAL (__builtin_huge_val())
#endif
