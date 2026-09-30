// EXPECT: 0x59
// REVOKE of a handle whose own node was revoked raises Invalid capability
// (25), as the spec and the RTL do. Before, the emulator walked the junior
// chain of the spliced-out node, rewrote its successor's prev link and handed
// back a LINEAR capability on a dead node.
#include "capstone.h"
TEST_PROLOGUE
    MINT_PAGE(s0, PAGE_A)
    MREV(s1, s0)
    MREV(s2, s0)                 // nested handle, junior to s1
    REVOKE(s1)                   // s2's node is now invalid
    EXPECT_FAULT_AT(site)
site:
    REVOKE(s2)
    FAIL(1)
