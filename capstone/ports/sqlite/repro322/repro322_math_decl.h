/* Declarations so the amalgamation (a separate TU) can call libm; definitions
 * are in repro322_fts_stubs.c. Injected via -include for FTS builds only. */
#ifndef REPRO322_MATH_DECL_H
#define REPRO322_MATH_DECL_H
double log(double); double log10(double); double log2(double);
double fabs(double); double floor(double); double ceil(double);
double sqrt(double); double pow(double,double); double exp(double);
/* qsort: FTS3 sorts Fts3HashElem* arrays for prefix queries; freestanding
 * build has no stdlib decl. Definition in repro322_fts_stubs.c. */
void qsort(void *, __SIZE_TYPE__, __SIZE_TYPE__, int (*)(const void *, const void *));
#endif
