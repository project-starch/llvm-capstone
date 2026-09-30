// EXPECT: 0x5b
// CREATE refuses a root page without write authority (cause 27): the page
// must be cleared, and later written by the walker's owner.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    TIGHTEN(s1, s1, PERM_RO)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    EXPECT_FAULT_AT(site)
site:
    MAPCREATE(s2, s1, s0)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
