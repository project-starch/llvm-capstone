// EXPECT: 0x5d
// CREATE refuses a range in the physical region (E1): logical ranges start
// at 2^57.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    li a0, 0
    li a1, 0x90000000
    li a2, 0x90100000
    li a3, 6
    li a4, 28
    EXPECT_FAULT_AT(site)
site:
    MAPCREATE(s2, s1, s0)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
