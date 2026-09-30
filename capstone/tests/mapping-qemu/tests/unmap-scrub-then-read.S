// EXPECT: 0
// The UNINIT frame from UNMAP becomes readable only after every granule was
// written: 256 null capability stores advance the cursor to the end, INIT
// yields a linear page, and the domain's bytes are gone.
#include "capstone.h"
TEST_PROLOGUE
    SETUP_MAPPING(6)
    POPULATE_TWO_PAGES
    CALL(s0, s0)
    MAPDETACH(s5, s2)
    li a0, 1
    slli a0, a0, 57
    MAPUNMAP(s6, s5)
    li t4, 256
2:  STC(s6, x0, 0)
    addi t4, t4, -1
    bnez t4, 2b
    INIT(s6, s6, x0)
    LCC(t0, s6, x1)
    bnez t0, 1f
    lw t0, 0(s6)
    bnez t0, 1f
    lw t0, 8(s6)
    bnez t0, 1f
    PASS
1:  FAIL(1)
_dom:
    li t0, 0x77
    sw t0, 0(t3)
    li t0, 0x99
    sw t0, 8(t3)
    DOM_RETURN(_dom)
_dom_end:
