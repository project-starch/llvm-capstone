// EXPECT: 0x59
// The domain delinearizes its mapping capability; DETACH still yields only
// the token (a REV, checked by the monitor), and the domain's non-linear copy
// is dead afterwards.
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
    DELIN(t3)
    MOVC(t5, t3)
    li t0, 0x77
    sw t0, 0(t5)
    DOM_RETURN(_dom_resume)
_dom_resume:
    EXPECT_FAULT_AT(site)
site:
    lw t0, 0(t5)
    FAIL(1)
_dom_end:
