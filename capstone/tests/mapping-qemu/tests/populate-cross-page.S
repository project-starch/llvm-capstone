// EXPECT: 0x46
// An access that straddles two logical pages is refused with a misaligned
// store fault (6) in this prototype rather than split across frames.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_E, 0, 20)
    POPULATE(s3, PAGE_G, 0x1000, 0)
    CALL(s0, s0)
_dom:
    li t0, 0x1000 - 4
    CINCOFFSET(t4, t3, t0)
    li t1, 1
    EXPECT_FAULT_AT(site)
site:
    sd t1, 0(t4)
    FAIL(1)
_dom_end:
