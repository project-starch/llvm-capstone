// EXPECT: 0x5d
// CREATE refuses a range that overlaps a reserved mapping, even one delivered
// to another domain (E1: global disjointness).
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    MAPCREATE(s2, s1, s0)
    MAKE_CONTEXT(s0, PAGE_D, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_B)
    CREATE_ARGS(1, 0x1000, 0x2000, 6, 28)
    EXPECT_FAULT_AT(site)
site:
    MAPCREATE(s2, s1, s0)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
