// EXPECT: 0
// ENV: CAPSTONE_REV_NODES=32
// Control: the reclamation loop without CREATE, DETACH and DESTROY.
#include "capstone.h"
TEST_PROLOGUE
    MINT_PAGE(s1, PAGE_A)
    li s8, 64
_loop:
    MREV(s7, s1)
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
