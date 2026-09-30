// EXPECT: 0
// A frame's supplier bytes and a stored capability are gone after POPULATE:
// the domain reads zero and an untagged granule (design §4, E6 of the plan).
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(t4, PAGE_E)
    li t0, 0x4242
    sw t0, 0(t4)
    MINT_PAGE(t5, PAGE_G)
    DELIN(t5)
    STC(t4, t5, 16)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_E, 0, 20)
    CALL(s0, s0)
_dom:
    lw t1, 0(t3)
    bnez t1, 1f
    LDC(t4, t3, 16)
    LCC(t5, t4, x1)
    li t6, 7
    bne t5, t6, 1f
    PASS
1:  FAIL(1)
_dom_end:
