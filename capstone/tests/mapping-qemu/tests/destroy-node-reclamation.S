// EXPECT: 0
// ENV: CAPSTONE_REV_NODES=32
// A CREATE, DETACH, DESTROY cycle with full reclamation of the root page
// leaves no live node behind: 64 rounds fit in a 32-node pool because DESTROY
// invalidates the mapping's senior node for the collector. Control:
// destroy-node-reclamation-control.S, the same loop without the mapping.
#include "capstone.h"
TEST_PROLOGUE
    MAKE_CONTEXT(s0, PAGE_C, _dom, _dom_end)
    MINT_PAGE(s1, PAGE_A)
    li s8, 64
_loop:
    MREV(s7, s1)
    CREATE_ARGS(0, 0, 0x100000, 6, 28)
    MAPCREATE(s2, s1, s0)
    MAPDETACH(s5, s2)
    MAPDESTROY(s5)
    REVOKE(s7)
    li t4, 256
2:  STC(s7, x0, 0)
    addi t4, t4, -1
    bnez t4, 2b
    INIT(s7, s7, x0)
    MOVC(s1, s7)
    SUPERVISOR_GC(t0)
    addi s8, s8, -1
    bnez s8, _loop
    PASS
_dom:
    FAIL(2)
_dom_end:
