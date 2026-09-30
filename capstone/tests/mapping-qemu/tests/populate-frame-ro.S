// EXPECT: 0x5b
// POPULATE refuses a frame without write authority (cause 27): it must be
// initialised, scrubbed on return, and carry the mapping's rights.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    MINT_PAGE(s3, PAGE_E)
    TIGHTEN(s3, s3, PERM_RO)
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
