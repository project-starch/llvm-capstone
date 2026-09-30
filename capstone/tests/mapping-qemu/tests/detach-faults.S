// EXPECT: 0x59
// After DETACH every logical capability of the mapping is dead: the domain's
// next access faults at its own revocation check (cause 25), while the
// monitor holds the token, a REV, not readable authority.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    POPULATE_TWO_PAGES
    CALL(s0, s0)
    MAPDETACH(s5, s2)
    LCC(t0, s5, x1)
    li t1, 2
    bne t0, t1, 1f
    CALL(s0, s0)
1:  FAIL(2)
_dom:
    li t0, 0x77
    sw t0, 0(t3)
    DOM_RETURN(_dom_resume)
_dom_resume:
    EXPECT_FAULT_AT(site)
site:
    lw t0, 0(t3)
    FAIL(1)
_dom_end:
