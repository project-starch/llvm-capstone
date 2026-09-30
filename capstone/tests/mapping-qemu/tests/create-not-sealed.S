// EXPECT: 0x5a
// CREATE refuses a recipient that is not a sealed context (cause 26): there
// is no way to deliver the mapping capability into a monitor register.
#include "capstone.h"
TEST_PROLOGUE
    MINT_PAGE(s0, PAGE_C)
    MINT_PAGE(s1, PAGE_A)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    EXPECT_FAULT_AT(site)
site:
    MAPCREATE(s2, s1, s0)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
