/* Hardware floating point in a virtual-C application: fenv is the hardware
 * fcsr, libm honours it, a thread starts with its creator's environment and
 * keeps its own, and a signal handler that returns leaves the interrupted
 * code's environment as it was.
 * Each passing check prints VIRTUAL_FP_OK <name>; run-fp.py counts them. */
#define _GNU_SOURCE
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The checks change and read the environment around FP arithmetic. */
#pragma STDC FENV_ACCESS ON

static int failures;
#define CHECK(cond) do { if (!(cond)) { \
    printf("VIRTUAL_FP_FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
    ++failures; } } while (0)

static uint64_t bits(double x)
{
    uint64_t b;
    memcpy(&b, &x, sizeof b);
    return b;
}

/* volatile operands: computed at run time, under the current rounding mode */
static volatile double one = 1.0, three = 3.0, ten = 10.0, zero = 0.0;
static volatile double huge_value = DBL_MAX, two_and_half = 2.5;

static double third(void) { return one / three; }
static double tenth(void) { return one / ten; }

static void report(const char *name, int before)
{
    if (failures == before) printf("VIRTUAL_FP_OK %s\n", name);
}

static void hardware(void)
{
    int before = failures;
    unsigned int fcsr;
    __asm__ volatile("frcsr %0" : "=r"(fcsr)); /* illegal unless FS is on */
    CHECK(fcsr == 0);
    CHECK(fegetround() == FE_TONEAREST);
    CHECK(fetestexcept(FE_ALL_EXCEPT) == 0);
    report("hardware_fresh_environment", before);
}

static void rounding(void)
{
    int before = failures;
    static const struct { int mode; uint64_t third, tenth; } want[] = {
        {FE_TONEAREST, 0x3fd5555555555555, 0x3fb999999999999a},
        {FE_UPWARD, 0x3fd5555555555556, 0x3fb999999999999a},
        {FE_DOWNWARD, 0x3fd5555555555555, 0x3fb9999999999999},
        {FE_TOWARDZERO, 0x3fd5555555555555, 0x3fb9999999999999},
    };
    for (unsigned i = 0; i < sizeof want / sizeof want[0]; ++i) {
        CHECK(fesetround(want[i].mode) == 0);
        CHECK(fegetround() == want[i].mode);
        CHECK(bits(third()) == want[i].third);
        CHECK(bits(tenth()) == want[i].tenth);
    }
    CHECK(fesetround(FE_TONEAREST) == 0);
    report("rounding_modes", before);
}

static void exceptions(void)
{
    int before = failures;
    volatile double r;
    CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
    r = one / zero;
    CHECK(fetestexcept(FE_ALL_EXCEPT) == FE_DIVBYZERO);
    CHECK(isinf(r));
    feclearexcept(FE_ALL_EXCEPT);
    r = zero / zero;
    CHECK(fetestexcept(FE_ALL_EXCEPT) == FE_INVALID);
    CHECK(isnan(r));
    feclearexcept(FE_ALL_EXCEPT);
    r = huge_value * ten;
    CHECK(fetestexcept(FE_ALL_EXCEPT) == (FE_OVERFLOW | FE_INEXACT));
    feclearexcept(FE_ALL_EXCEPT);
    r = third();
    CHECK(fetestexcept(FE_ALL_EXCEPT) == FE_INEXACT);
    feclearexcept(FE_ALL_EXCEPT);
    long converted;
    double big = huge_value;
    __asm__ volatile("fcvt.l.d %0, %1, rtz" : "=r"(converted) : "f"(big)); /* out of range */
    CHECK(converted == LONG_MAX);
    CHECK(fetestexcept(FE_ALL_EXCEPT) == FE_INVALID);
    feclearexcept(FE_ALL_EXCEPT);
    CHECK(feraiseexcept(FE_UNDERFLOW) == 0);
    CHECK(fetestexcept(FE_ALL_EXCEPT) == FE_UNDERFLOW);
    /* environment round trip, the default environment, hold/update */
    fenv_t saved, held;
    fesetround(FE_DOWNWARD);
    CHECK(fegetenv(&saved) == 0);
    CHECK(fesetenv(FE_DFL_ENV) == 0);
    CHECK(fegetround() == FE_TONEAREST && fetestexcept(FE_ALL_EXCEPT) == 0);
    CHECK(fesetenv(&saved) == 0);
    CHECK(fegetround() == FE_DOWNWARD && fetestexcept(FE_ALL_EXCEPT) == FE_UNDERFLOW);
    CHECK(feholdexcept(&held) == 0);
    CHECK(fetestexcept(FE_ALL_EXCEPT) == 0);
    r = one / zero;
    CHECK(feupdateenv(&held) == 0);
    CHECK(fetestexcept(FE_ALL_EXCEPT) == (FE_UNDERFLOW | FE_DIVBYZERO));
    CHECK(fegetround() == FE_DOWNWARD);
    fesetenv(FE_DFL_ENV);
    report("exception_flags_and_environment", before);
}

static void libm(void)
{
    int before = failures;
    CHECK(bits(sqrt(2.0 * one)) == 0x3ff6a09e667f3bcd);
    CHECK(fabs(exp(one) - 2.718281828459045) < 1e-15);
    CHECK(fabs(log(10.0 * one) - 2.302585092994046) < 1e-15);
    CHECK(fabs(sin(one) - 0.8414709848078965) < 1e-15);
    CHECK(pow(2.0 * one, 10.0) == 1024.0);
    CHECK(fma(one / three, three, -one) != 0.0); /* fused: the rounding error of 1/3 */
    char text[64];
    snprintf(text, sizeof text, "%.17g", third());
    CHECK(bits(strtod(text, 0)) == bits(third()));
    /* rint and nearbyint follow the dynamic rounding mode; nearbyint
     * leaves the inexact flag as it was. */
    CHECK(rint(two_and_half) == 2.0);
    fesetround(FE_UPWARD);
    CHECK(rint(two_and_half) == 3.0 && lrint(two_and_half) == 3);
    feclearexcept(FE_ALL_EXCEPT);
    CHECK(nearbyint(two_and_half) == 3.0);
    CHECK(fetestexcept(FE_INEXACT) == 0);
    fesetround(FE_TOWARDZERO);
    CHECK(rint(two_and_half) == 2.0 && rint(-two_and_half) == -2.0);
    fesetenv(FE_DFL_ENV);
    float f = (float)one / (float)three;
    CHECK(f == 0x1.555556p-2f);
    /* long double stays soft (binary128); its addition follows the same
     * environment, and so does rintl, which is built on it (compiler-rt's
     * multiplication, division and narrowing still round to nearest) */
    volatile long double lone = 1.0L, ltiny = 0x1p-120L;
    feclearexcept(FE_ALL_EXCEPT);
    CHECK(lone + ltiny == 1.0L);
    CHECK(fetestexcept(FE_ALL_EXCEPT) == FE_INEXACT);
    fesetround(FE_UPWARD);
    CHECK(lone + ltiny == 1.0L + 0x1p-112L);
    CHECK(rintl(2.5L * lone) == 3.0L);
    fesetenv(FE_DFL_ENV);
    report("libm_follows_hardware", before);
}

/* A thread starts with its creator's environment, then keeps its own: the
 * two hand a turn back and forth, 200 times, each checking its own rounding
 * mode and flags on every turn, so each check follows a switch away from the
 * other thread's environment. */
#define TURNS 200
static volatile int turn; /* 0: creator, 1: child */
static void *keeps_own(void *arg)
{
    int *ok = arg;
    *ok = fegetround() == FE_UPWARD && fetestexcept(FE_ALL_EXCEPT) == FE_INEXACT;
    fesetround(FE_DOWNWARD);
    feclearexcept(FE_ALL_EXCEPT);
    feraiseexcept(FE_OVERFLOW);
    for (unsigned i = 0; i < TURNS; ++i) {
        while (__atomic_load_n(&turn, __ATOMIC_ACQUIRE) != 1) sched_yield();
        if (*ok)
            *ok = fegetround() == FE_DOWNWARD && fetestexcept(FE_ALL_EXCEPT) == FE_OVERFLOW &&
                  bits(tenth()) == 0x3fb9999999999999;
        feclearexcept(FE_INEXACT);
        __atomic_store_n(&turn, 0, __ATOMIC_RELEASE);
    }
    return 0;
}

static void threads(void)
{
    int before = failures, ok = 0, mine = 1;
    pthread_t t;
    fesetround(FE_UPWARD);
    feclearexcept(FE_ALL_EXCEPT);
    feraiseexcept(FE_INEXACT);
    turn = 0;
    CHECK(pthread_create(&t, 0, keeps_own, &ok) == 0);
    feclearexcept(FE_ALL_EXCEPT); /* after creation: the child keeps INEXACT */
    feraiseexcept(FE_DIVBYZERO);
    for (unsigned i = 0; i < TURNS; ++i) {
        while (__atomic_load_n(&turn, __ATOMIC_ACQUIRE) != 0) sched_yield();
        if (mine)
            mine = fegetround() == FE_UPWARD && fetestexcept(FE_ALL_EXCEPT) == FE_DIVBYZERO &&
                   bits(tenth()) == 0x3fb999999999999a;
        feclearexcept(FE_INEXACT);
        __atomic_store_n(&turn, 1, __ATOMIC_RELEASE);
    }
    CHECK(pthread_join(t, 0) == 0);
    CHECK(mine);
    CHECK(ok);
    CHECK(fegetround() == FE_UPWARD);
    fesetenv(FE_DFL_ENV);
    report("thread_inherits_then_keeps_its_own", before);
}

static volatile int handler_saw, handler_ran;
static void handler(int sig)
{
    (void)sig;
    handler_saw = fegetround();
    fesetround(FE_TOWARDZERO);
    feraiseexcept(FE_OVERFLOW);
    handler_ran = 1;
}

static void signals(void)
{
    int before = failures;
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = handler;
    CHECK(sigaction(SIGUSR1, &sa, 0) == 0);
    fesetround(FE_UPWARD);
    feclearexcept(FE_ALL_EXCEPT);
    CHECK(raise(SIGUSR1) == 0);
    CHECK(handler_ran);
    CHECK(handler_saw == FE_UPWARD); /* the handler starts with the interrupted environment */
    CHECK(fegetround() == FE_UPWARD); /* and its changes end with it */
    CHECK(fetestexcept(FE_ALL_EXCEPT) == 0);
    fesetenv(FE_DFL_ENV);
    report("signal_handler_leaves_environment", before);
}

/* modf and frexp return one result through a pointer. The compiler expands
 * modf (and frexp, when it sees the intrinsic) into a call that writes a stack
 * temporary; that pointer used to be passed as an integer, and modf faulted
 * storing the integral part (C-86). Last, so a fault leaves the rest reported.
 * FENV_ACCESS is off here: under it clang keeps modf a plain call, which passes
 * the caller's pointer through and never reaches that expansion. */
#pragma STDC FENV_ACCESS OFF
static void split(void)
{
    int before = failures;
    volatile double x = 3.75 * one;
    double ip = 0, *ipp = &ip;
    CHECK(modf(x, &ip) == 0.75 && ip == 3.0);
    CHECK(modf(-x, ipp) == -0.75 && ip == -3.0);
    float fip = 0;
    CHECK(modff((float)x, &fip) == 0.75f && fip == 3.0f);
    long double lip = 0;
    CHECK(modfl((long double)x, &lip) == 0.75L && lip == 3.0L);
    int e = 0;
    CHECK(frexp(x, &e) == 0.9375 && e == 2);
    CHECK(frexpl((long double)x, &e) == 0.9375L && e == 2);
    report("split_results_through_pointers", before);
}
#pragma STDC FENV_ACCESS ON

int main(void)
{
    hardware();
    rounding();
    exceptions();
    libm();
    threads();
    signals();
    split();
    printf("VIRTUAL_FP_DONE failures=%d\n", failures);
    return failures != 0;
}
