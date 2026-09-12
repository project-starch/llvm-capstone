/* The two libm entry points MicroPython's own lib/libm_dbl does not ship a file for. */

#include <stdint.h>

/* pow.c and objfloat.c both call fabs; -ffreestanding means it is a real call, not a builtin. */
double fabs(double x) {
    return __builtin_fabs(x);
}

double nan(const char *tagp) {
    (void)tagp;
    union {
        uint64_t bits;
        double value;
    } result = {.bits = UINT64_C(0x7ff8000000000000)};
    return result.value;
}
