// EXPECT: 0x5a
// A second UNMAP of the same page finds the slot locked (cause 26).
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    POPULATE_TWO_PAGES
    MAPDETACH(s5, s2)
    li a0, 1
    slli a0, a0, 57
    MAPUNMAP(s6, s5)
    EXPECT_FAULT_AT(site)
site:
    MAPUNMAP(s7, s5)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
