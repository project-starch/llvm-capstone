/* SPDX-License-Identifier: MIT */
#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

_Static_assert(sizeof(struct timespec)==16, "RV64 timespec wire size");
_Static_assert(offsetof(struct timespec,tv_sec)==0 && offsetof(struct timespec,tv_nsec)==8, "RV64 timespec wire offsets");
static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr,"CLOCK_FAIL line=%d errno=%d\n",__LINE__,errno); } } while (0)
static int normalized(struct timespec t) {
    return t.tv_sec >= 0 && t.tv_nsec >= 0 && t.tv_nsec < 1000000000;
}
static int ordered(struct timespec a, struct timespec b) {
    return b.tv_sec > a.tv_sec || (b.tv_sec == a.tv_sec && b.tv_nsec >= a.tv_nsec);
}
#ifdef TIME_RAW
extern long __capstone_native_syscall(long, void *, void *, void *, void *, void *, void *);
static void *scalar(unsigned long n) {
    void *r;
    __asm__ volatile("movc %0, zero\n\tmv %0, %1" : "=&r"(r) : "r"(n));
    return r;
}
/* Keep malformed capabilities intact while exposing the ordinary errno oracle. */
static long raw_clock(void *id, void *output, void *extra) {
    long rc=__capstone_native_syscall(113,id,output,extra,0,0,0);
    if (rc<0) { errno=-rc; return -1; }
    return rc;
}
static void raw_checks(void) {
    struct { uint64_t before; struct timespec time; uint64_t after; } box = {
        0x123456789abcdef0ULL, {123,456}, 0xfedcba9876543210ULL
    };
    unsigned long base=__builtin_capstone_cap_get_cursor(&box.time);
    void *tight=__builtin_capstone_cap_shrink(&box.time,base,base+16);
    void *short_output=__builtin_capstone_cap_shrink(&box.time,base,base+15);
    void *readonly=__builtin_capstone_cap_tighten(tight,4);
    void *writeonly=__builtin_capstone_cap_tighten(tight,2);
    void *bad_outputs[]={0,short_output,readonly,scalar(base)};
    _Alignas(16) unsigned long words[2]={CLOCK_MONOTONIC,1};
    void *dirty;
    __asm__ volatile("ldc %0, 0(%1)" : "=r"(dirty) : "r"(words) : "memory");
#define UNTOUCHED (box.time.tv_sec==123 && box.time.tv_nsec==456 && box.before==0x123456789abcdef0ULL && box.after==0xfedcba9876543210ULL)
    errno=0; CHECK(raw_clock(scalar(777),tight,0)==-1 && errno==EINVAL && UNTOUCHED);
    errno=0; CHECK(raw_clock(scalar(~0UL),tight,0)==-1 && errno==EINVAL && UNTOUCHED);
    errno=0; CHECK(raw_clock(scalar((1UL<<32)|CLOCK_MONOTONIC),tight,0)==-1 && errno==EINVAL && UNTOUCHED);
    for (unsigned i=0;i<sizeof(bad_outputs)/sizeof(bad_outputs[0]);++i) {
        errno=0; CHECK(raw_clock(scalar(CLOCK_MONOTONIC),bad_outputs[i],0)==-1 && errno==EFAULT && UNTOUCHED);
        /* Preserve this provider's output-authority-before-ID precedence. */
        errno=0; CHECK(raw_clock(scalar(777),bad_outputs[i],0)==-1 && errno==EFAULT && UNTOUCHED);
    }
    errno=0; CHECK(raw_clock(tight,tight,0)==-1 && errno==EINVAL && UNTOUCHED);
    errno=0; CHECK(raw_clock(scalar(CLOCK_MONOTONIC),tight,tight)==-1 && errno==EINVAL && UNTOUCHED);
    errno=0; CHECK(raw_clock(dirty,tight,0)==-1 && errno==EINVAL && UNTOUCHED);
    CHECK(raw_clock(scalar(CLOCK_MONOTONIC),tight,0)==0 && normalized(box.time) && box.before==0x123456789abcdef0ULL && box.after==0xfedcba9876543210ULL);
    CHECK(raw_clock(scalar(CLOCK_REALTIME),writeonly,0)==0 && normalized(box.time) && box.before==0x123456789abcdef0ULL && box.after==0xfedcba9876543210ULL);
#undef UNTOUCHED
}
#endif
int main(void) {
    struct timespec before={0},after={0},real={0},bad={123,456};
    CHECK(sizeof(struct timespec)==16 && sizeof(time_t)==8);
    CHECK(offsetof(struct timespec,tv_nsec)==8);
    CHECK(clock_gettime(CLOCK_MONOTONIC,&before)==0 && normalized(before));
    CHECK(clock_gettime(CLOCK_MONOTONIC,&after)==0 && normalized(after));
    CHECK(ordered(before,after));
    CHECK(clock_gettime(CLOCK_REALTIME,&real)==0 && normalized(real) && real.tv_sec>0);
    const int unsupported[]={777,-1,2,3,4,5,6,7};
    for (unsigned i=0;i<sizeof unsupported/sizeof *unsupported;++i) {
        errno=0; CHECK(clock_gettime(unsupported[i],&bad)==-1 && errno==EINVAL && bad.tv_sec==123 && bad.tv_nsec==456);
    }
#ifdef TIME_RAW
    raw_checks();
#endif
    fprintf(stderr,"NATIVE_CLOCK checks=%d failures=%d\n",checks,failures);
    return failures!=0;
}
