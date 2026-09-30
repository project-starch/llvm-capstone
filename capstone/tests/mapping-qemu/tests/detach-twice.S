// EXPECT: 0x5a
// DETACH refuses the token of a mapping already DETACHED (cause 26).
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    POPULATE_TWO_PAGES
    MAPDETACH(s5, s2)
    EXPECT_FAULT_AT(site)
site:
    MAPDETACH(s6, s5)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
