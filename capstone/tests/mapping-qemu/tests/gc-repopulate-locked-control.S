// EXPECT: 0x5a
// Control for gc-repopulate-locked: the same sequence without the collector.
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
    REVOKE(s7)
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
