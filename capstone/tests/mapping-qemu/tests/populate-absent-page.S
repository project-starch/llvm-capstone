// EXPECT: 0x45
// A load from a page inside the mapping that was never populated faults with
// a load access fault (5); it is not allocated on demand.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_E, 0, 20)
    CALL(s0, s0)
_dom:
    li t0, 0x1000
    CINCOFFSET(t4, t3, t0)
    EXPECT_FAULT_AT(site)
site:
    lw t1, 0(t4)
    FAIL(1)
_dom_end:
