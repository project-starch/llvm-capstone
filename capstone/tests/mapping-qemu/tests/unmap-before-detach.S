// EXPECT: 0x5a
// UNMAP refuses the detach handle of an ACTIVE mapping (cause 26): frames
// leave a live mapping only through revocation from above.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    POPULATE_TWO_PAGES
    li a0, 1
    slli a0, a0, 57
    EXPECT_FAULT_AT(site)
site:
    MAPUNMAP(s6, s2)
    FAIL(1)
_dom:
    FAIL(2)
_dom_end:
