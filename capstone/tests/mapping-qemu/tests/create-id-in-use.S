// EXPECT: 0x5e
// CREATE refuses an id whose registry entry is in use (cause 30), even for a
// disjoint range and another recipient.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    MAPCREATE(s2, s1, s0)
    MAKE_CONTEXT(s0, PAGE_D, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_B)
    CREATE_ARGS(0, 0x200000, 0x1000, 6, 28)
    EXPECT_FAULT_AT(site)
site:
    MAPCREATE(s2, s1, s0)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
