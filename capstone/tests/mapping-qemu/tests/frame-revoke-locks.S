// EXPECT: 0x59
// Revoking a mapped frame's handle from above returns UNINIT to the holder,
// leaves the other page live, and makes the domain's access to that page
// fault (cause 25); the PTE is locked, not refillable.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    MINT_PAGE(s3, PAGE_E)
    MREV(s7, s3)
    li a0, 1
    slli a0, a0, 57
    li a5, 20
    MAPPOPULATE(s2, s3)
    POPULATE(s3, PAGE_G, 0x1000, 0)
    CALL(s0, s0)
    REVOKE(s7)
    LCC(t0, s7, x1)
    li t1, 3
    bne t0, t1, 1f
    CALL(s0, s0)
1:  FAIL(2)
_dom:
    li t0, 0x77
    sw t0, 0(t3)
    DOM_RETURN(_dom_resume)
_dom_resume:
    li t0, 0x1000
    DELIN(t3)
    CINCOFFSET(t4, t3, t0)
    li t0, 0x55
    sw t0, 0(t4)
    lw t1, 0(t4)
    bne t0, t1, 1f
    EXPECT_FAULT_AT(site)
site:
    lw t0, 0(t3)
1:  FAIL(1)
_dom_end:
