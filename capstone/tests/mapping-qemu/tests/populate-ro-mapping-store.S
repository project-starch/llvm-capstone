// EXPECT: 0x5b
// A store through the capability of a read-only mapping faults on permission
// (27); the mapping's max protection is the ceiling. Control:
// populate-ro-mapping-load.S.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(4)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_E, 0, 20)
    CALL(s0, s0)
_dom:
    li t0, 1
    EXPECT_FAULT_AT(site)
site:
    sw t0, 0(t3)
    FAIL(1)
_dom_end:
