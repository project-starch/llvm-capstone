// EXPECT: 0x5a
// POPULATE refuses a PTE that is not none (cause 26): a used slot never
// changes its target within a generation (invariant I5).
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    POPULATE(s3, PAGE_E, 0, 20)
    MINT_PAGE(s3, PAGE_G)
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
