// EXPECT: 0x5a
// DESTROY refuses the detach handle of an ACTIVE mapping (cause 26).
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    POPULATE_TWO_PAGES
    EXPECT_FAULT_AT(site)
site:
    MAPDESTROY(s2)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
