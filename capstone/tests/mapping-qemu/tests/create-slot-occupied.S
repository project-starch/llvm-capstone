// EXPECT: 0x5d
// CREATE refuses a recipient whose delivery slot still holds an undelivered
// mapping capability: delivery is exactly once, never overwritten.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    MAPCREATE(s2, s1, s0)
    MINT_PAGE(s1, PAGE_B)
    CREATE_ARGS(1, 0x200000, 0x1000, 6, 29)
    EXPECT_FAULT_AT(site)
site:
    MAPCREATE(s2, s1, s0)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
