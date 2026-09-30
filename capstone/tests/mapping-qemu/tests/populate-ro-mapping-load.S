// EXPECT: 0
// Control: a load through the capability of a read-only mapping reads the
// zero-filled frame.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(4)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_E, 0, 20)
    CALL(s0, s0)
_dom:
    lw t0, 0(t3)
    bnez t0, 1f
    PASS
1:  FAIL(1)
_dom_end:
