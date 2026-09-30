// EXPECT: 0x5a
// UNMAP returns the frame as UNINIT: a load through it before it is written
// faults (cause 26), so the monitor cannot read what the domain left.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    POPULATE_TWO_PAGES
    CALL(s0, s0)
    MAPDETACH(s5, s2)
    li a0, 1
    slli a0, a0, 57
    MAPUNMAP(s6, s5)
    LCC(t0, s6, x1)
    li t1, 3
    bne t0, t1, 1f
    EXPECT_FAULT_AT(site)
site:
    lw t0, 0(s6)
1:  FAIL(1)
_dom:
    li t0, 0x77
    sw t0, 0(t3)
    DOM_RETURN(_dom)
_dom_end:
