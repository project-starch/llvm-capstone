// EXPECT: 0
// An atomic read-modify-write on logical memory goes through the same
// translation: amoadd returns the old value and leaves the sum.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_E, 0, 20)
    CALL(s0, s0)
_dom:
    li t0, 5
    sw t0, 0(t3)
    li t1, 3
    amoadd.w t2, t1, (t3)
    li t0, 5
    bne t2, t0, 1f
    lw t1, 0(t3)
    li t0, 8
    bne t1, t0, 1f
    PASS
1:  FAIL(1)
_dom_end:
