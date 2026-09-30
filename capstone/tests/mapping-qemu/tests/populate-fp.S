// EXPECT: 0
// Floating-point loads and stores through a logical capability are checked
// and translated like integer accesses.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_E, 0, 20)
    CALL(s0, s0)
_dom:
    li t0, 1 << 13
    csrs mstatus, t0
    li t0, 0x3ff0000000000000
    sd t0, 0(t3)
    fld f0, 0(t3)
    fsd f0, 8(t3)
    ld t1, 8(t3)
    bne t0, t1, 1f
    PASS
1:  FAIL(1)
_dom_end:
