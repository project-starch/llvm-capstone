// EXPECT: 0x5d
// POPULATE refuses a frame that is not writable RAM (virt's ROM page at
// 0x1000), before any effect: a page that cannot be cleared is never
// published. On 0efac317 the ROM page was accepted and the domain read its
// bytes.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    MINT_PAGE(s4, PAGE_F)
    MINT_PAGE(s3, 0x1000)
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
