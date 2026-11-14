/* inject.h: test-only fault injection for the lifetime patches. Force-included (-include) into the
 * allocator's translation unit of a unit-test build, never into a product build. It wraps CDERIVE
 * (capstone_cap_derive) and CRENEW (capstone_cap_renew): while lifec_fail_left > 0, a call whose
 * size matches lifec_fail_size (0 matches every size; CRENEW counts as size 0) returns the untagged
 * 0 that the instruction returns when the lifetime table has no slot and cannot grow, and changes
 * nothing. The test program defines the four counters. */
#ifndef LIFEC_INJECT_H
#define LIFEC_INJECT_H
#include <capstone/capability.h>
extern long lifec_fail_left, lifec_fail_size, lifec_failed;
extern unsigned long lifec_failed_at;   /* the address the last failed CDERIVE would have started at */
static inline int lifec_fail(unsigned long size) {
    if (lifec_fail_left > 0 && (lifec_fail_size == 0 || (unsigned long)lifec_fail_size == size)) {
        lifec_fail_left--;
        lifec_failed++;
        return 1;
    }
    return 0;
}
static inline void *lifec_derive(void *parent, unsigned long offset, unsigned long size) {
    if (lifec_fail(size)) {
        lifec_failed_at = __builtin_capstone_cap_get_base(parent) + offset;
        return (void *)0;
    }
    return capstone_cap_derive(parent, offset, size);
}
static inline void *lifec_renew(void *object) {
    return lifec_fail(0) ? (void *)0 : capstone_cap_renew(object);
}
#define capstone_cap_derive lifec_derive
#define capstone_cap_renew lifec_renew
#endif
