// EXPECT: 0x5d
// POPULATE refuses, consuming nothing, when the path needs a leaf table and
// none was supplied (cause 29); the frame is still usable afterwards.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s3, PAGE_E)
    li a0, 1
    slli a0, a0, 57
    li a5, 0
    EXPECT_FAULT_AT(site)
site:
    MAPPOPULATE(s2, s3)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
