// EXPECT: 0x5a
// POPULATE refuses a non-linear frame (cause 26): PRIVATE takes exclusive
// backing only (invariant I4).
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    MINT_PAGE(s3, PAGE_E)
    DELIN(s3)
    li a0, 1
    slli a0, a0, 57
    li a5, 20
    EXPECT_FAULT_AT(site)
site:
    MAPPOPULATE(s2, s3)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
