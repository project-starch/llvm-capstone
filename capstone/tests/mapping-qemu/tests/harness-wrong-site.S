// EXPECT: 0x3f
// Harness self-check: a fault before the named site exits with WRONG_SITE,
// so a refusal test that faults in its setup cannot pass by accident.
#include "capstone.h"
TEST_PROLOGUE
    MINT_PAGE(t2, PAGE_A)
    TIGHTEN(t2, t2, PERM_RO)
    EXPECT_FAULT_AT(site)
    li t3, 1
    sw t3, 0(t2)
site:
    sw t3, 4(t2)
    FAIL(1)
