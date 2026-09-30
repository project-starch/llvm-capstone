// EXPECT: 0
// lcc selector 0 reads the node's validity (spec LCC table, RTL
// send_node_query_validity): 1 for a live capability and handle, 0 for a
// delinearized alias and for a nested handle after their senior handle was
// revoked. The libc checks it before revoking a mapping it may already have
// lost to revocation from above.
#include "capstone.h"
TEST_PROLOGUE
    MINT_PAGE(s0, PAGE_A)
    MREV(s1, s0)                 // s1: handle senior to s0
    LCC(t3, s1, x0)
    li t4, 1
    bne t3, t4, 1f
    MREV(s2, s0)                 // s2: nested handle, junior to s1
    LCC(t3, s2, x0)
    bne t3, t4, 2f
    DELIN(s0)
    MOVC(s3, s0)                 // s3: a copy of the delinearized alias
    LCC(t3, s3, x0)
    bne t3, t4, 3f
    REVOKE(s1)
    LCC(t3, s3, x0)              // the alias died with s1's juniors
    bnez t3, 4f
    LCC(t3, s2, x0)              // so did the nested handle
    bnez t3, 5f
    LCC(t3, s1, x0)              // the revoking handle itself stays valid
    bne t3, t4, 6f
    PASS
1:  FAIL(1)
2:  FAIL(2)
3:  FAIL(3)
4:  FAIL(4)
5:  FAIL(5)
6:  FAIL(6)
