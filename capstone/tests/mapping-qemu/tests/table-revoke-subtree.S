// EXPECT: 0x59
// Revoking a leaf table page's handle after the domain warmed the translation
// cache makes every page under it fault (cause 25): the flush on REVOKE and
// the walker's check of every node on the path.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    MREV(s7, s4)
    POPULATE(s3, PAGE_E, 0, 20)
    POPULATE(s3, PAGE_G, 0x1000, 0)
    CALL(s0, s0)
    REVOKE(s7)
    LCC(t0, s7, x1)
    li t1, 3
    bne t0, t1, 1f
    CALL(s0, s0)
1:  FAIL(2)
_dom:
    DELIN(t3)
    li t0, 0x77
    sw t0, 0(t3)
    li t0, 0x1000
    CINCOFFSET(t4, t3, t0)
    sw t0, 0(t4)
    DOM_RETURN(_dom_resume)
_dom_resume:
    EXPECT_FAULT_AT(site)
site:
    lw t0, 0(t4)
    FAIL(1)
_dom_end:
